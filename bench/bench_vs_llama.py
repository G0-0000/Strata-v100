#!/usr/bin/env python3
"""Strata vs llama.cpp head-to-head (stream 1402-h).  Measurement only.

Both engines are driven through their OpenAI-compatible HTTP servers (/v1/chat/completions), with the same prompt
text, same sampler settings and the same GGUF.  Server-side timings (llama.cpp's names, which Strata's server also
emits) give prompt tok/s and decode tok/s; time to first token is the server's prompt_ms (non-streaming request,
so TTFT = prompt processing time).  Peak VRAM comes from nvidia-smi polling (minus the idle baseline), RAM from the
engine process tree (psutil RSS, which counts mmap'd file pages) and the system's used-memory delta.

    python bench_vs_llama.py prompts  --out DIR --tokenizer DIR --corpus FILE...     # build 4K / 32K prompt files
    python bench_vs_llama.py run      --machine CFG.json --out DIR --rounds N [--configs a,b] [--cells ...]
    python bench_vs_llama.py quality  --machine CFG.json --out DIR [--configs a,b]
    python bench_vs_llama.py summary  --out DIR

CFG.json: {"gguf": ..., "strata_cfg": ..., "llama_server": ..., "configs": [{"name","kind":"llama|strata","args":[...]}]}
Interleaving: engines cannot be loaded together (12 GB card, 64 GB RAM), so one ROUND runs every config once
(start, warm-up, cells, stop), and rounds alternate the config order.  Medians are taken over rounds.
The caller holds the machine's GPU lock.  Processes are stopped by PID only.
"""
import argparse, copy, json, os, re, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

import psutil

CELLS = [("4k", 4000, 2), ("32k", 32000, 1)]       # (name, prompt tokens, reps per round)
TEMPS = [0.0, 0.7]
GEN = 256
PORT = 8190
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def valid_name(name):
    return (isinstance(name, str) and bool(name) and name not in (".", "..")
            and not any(ch in name for ch in ("/", "\\")) and not any(ord(ch) < 32 for ch in name))


def client_key(env_key, scfg, kind):
    raw = env_key or (scfg.get("api_key") if kind == "strata" and scfg else None)
    if isinstance(raw, list):
        return str(raw[0]).strip() if raw else None
    if not raw or str(raw).strip().lower() == "none":
        return None
    return next((part.strip() for part in str(raw).split(",") if part.strip()), None)


def api_secrets(env):
    value = env.get("STRATA_API_KEY", "")
    if not value or value.strip().lower() == "none":
        return []
    return [value] + [part.strip() for part in value.split(",") if part.strip()]


def redact(text, secrets):
    for secret in secrets:
        text = text.replace(secret, "<redacted>")
    return text


def safe_command(cmd, secrets):
    parts, hide_next = [], False
    for part in cmd:
        if hide_next:
            parts.append("<redacted>")
            hide_next = False
        elif part == "--api-key":
            parts.append(part)
            hide_next = True
        elif part.startswith("--api-key="):
            parts.append("--api-key=<redacted>")
        else:
            parts.append(redact(part, secrets))
    return " ".join(parts)


def load_machine(a):
    if not a.machine:
        raise ValueError("run 和 quality 需要 --machine")
    with open(a.machine, encoding="utf-8-sig") as f:
        m = json.load(f)
    if not isinstance(m, dict) or not isinstance(m.get("configs"), list) or not m["configs"]:
        raise ValueError("机器配置必须包含非空 configs 数组")
    names = set()
    for c in m["configs"]:
        if not isinstance(c, dict) or not valid_name(c.get("name")):
            raise ValueError("配置名称不能为空或包含目录分隔符、控制字符")
        if c["name"] in names:
            raise ValueError("配置名称重复：%s" % c["name"])
        names.add(c["name"])
        if c.get("kind") not in ("strata", "llama"):
            raise ValueError("%s 的 kind 必须是 strata 或 llama" % c["name"])
        if not isinstance(c.get("args"), list) or any(not isinstance(x, str) or "\0" in x for x in c["args"]):
            raise ValueError("%s 的 args 必须是字符串数组" % c["name"])
        env = c.get("env", {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not ENV_NAME.fullmatch(k)
                                            or not isinstance(v, str) or "\0" in v for k, v in env.items()):
            raise ValueError("%s 的 env 必须由合法变量名和字符串值组成" % c["name"])
        override = c.get("override", {})
        if not isinstance(override, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                  for k, v in override.items()):
            raise ValueError("%s 的 override 必须由字符串键和值组成" % c["name"])
    selected = set(a.configs.split(",")) if a.configs else names
    if not selected or selected - names:
        raise ValueError("--configs 包含未知配置名称")
    cfgs = [c for c in m["configs"] if c["name"] in selected]
    for key in ("cwd", "tokenizer", "strata_exe"):
        if key in m and (not isinstance(m[key], str) or not m[key] or "\0" in m[key]):
            raise ValueError("机器配置的 %s 必须是非空路径" % key)
    kinds = {c["kind"] for c in (m["configs"] if a.require_env_file or a.check else cfgs)}
    for key in (["strata_cfg", "serve"] if "strata" in kinds else []) + (
            ["llama_server", "gguf"] if "llama" in kinds else []):
        if not isinstance(m.get(key), str) or not m[key]:
            raise ValueError("机器配置缺少有效的 %s" % key)
    if "strata" in kinds:
        with open(m["strata_cfg"], encoding="utf-8-sig") as f:
            scfg = json.load(f)
        if not isinstance(scfg, dict) or not isinstance(scfg.get("args"), list) or any(
                not isinstance(x, str) or "\0" in x for x in scfg["args"]):
            raise ValueError("strata_cfg 中的 args 必须是字符串数组")
        source_env = scfg.get("env") or {}
        if not isinstance(source_env, dict) or any(not isinstance(k, str) or not ENV_NAME.fullmatch(k)
                                                   or "\0" in str(v) for k, v in source_env.items()):
            raise ValueError("strata_cfg 中的 env 无效")
        for c in m["configs"]:
            if c["kind"] == "strata":
                for flag in c.get("override", {}):
                    if flag in scfg["args"] and scfg["args"].index(flag) == len(scfg["args"]) - 1:
                        raise ValueError("%s 的 override 无法替换缺少值的参数" % c["name"])
        m["_strata_cfg_data"] = scfg
    cells = CELLS
    if "cells" in m:
        raw_cells = m["cells"]
        if not isinstance(raw_cells, list) or not raw_cells:
            raise ValueError("cells 必须是非空数组")
        cells, cell_names = [], set()
        for cell in raw_cells:
            if not isinstance(cell, dict) or not valid_name(cell.get("name")):
                raise ValueError("cell 名称无效")
            if cell["name"] in cell_names or type(cell.get("prompt_tokens")) is not int or cell["prompt_tokens"] < 1 \
                    or type(cell.get("reps")) is not int or cell["reps"] < 1:
                raise ValueError("cell 名称重复，或 prompt_tokens / reps 不是正整数")
            cell_names.add(cell["name"])
            cells.append((cell["name"], cell["prompt_tokens"], cell["reps"]))
    if type(m.get("gen", GEN)) is not int or m.get("gen", GEN) < 1:
        raise ValueError("gen 必须是正整数")
    selected_cells = set(a.cells.split(",")) if a.cells else {c[0] for c in cells}
    if not selected_cells or selected_cells - {c[0] for c in cells}:
        raise ValueError("--cells 包含未知名称")
    return m, cfgs, [c for c in cells if c[0] in selected_cells]


def read_required_env(path):
    required, file_vars = {}, set()
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if not sep or not ENV_NAME.fullmatch(name) or name in required:
            raise ValueError("必需环境文件第 %d 行格式无效或变量重复" % number)
        if value.startswith("<file:"):
            if not value.endswith(">") or not Path(value[6:-1]).is_absolute():
                raise ValueError("必需环境文件第 %d 行需要绝对文件路径" % number)
            try:
                value = Path(value[6:-1]).read_text(encoding="utf-8").rstrip("\r\n")
            except (OSError, UnicodeError, ValueError):
                raise ValueError("无法读取必需环境文件第 %d 行指定的值文件" % number) from None
            file_vars.add(name)
        if "\0" in value:
            raise ValueError("必需环境文件第 %d 行的值含有 NUL" % number)
        required[name] = value
    if not required:
        raise ValueError("必需环境文件为空")
    return required, file_vars


def sensitive_name(name):
    return any(part in name.upper() for part in ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH"))


def env_summary(cfgs, engine_envs, required, file_vars, allowed, source_env):
    names = set(required) | set(source_env)
    for cfg in cfgs:
        names.update(cfg.get("env", {}))
    secrets = [value for name, value in required.items() if name in file_vars or sensitive_name(name)]
    for env in engine_envs.values():
        secrets.extend(value for name, value in env.items() if sensitive_name(name))
    secrets = [value for value in secrets if value]
    first = engine_envs[cfgs[0]["name"]]
    for cfg in cfgs:
        name = cfg["name"]
        env = engine_envs[name]
        print("配置 %s 的有效环境（与首项比较）：" % name)
        for var in sorted(names):
            value = env.get(var)
            if value is None:
                shown = "<missing>"
            elif var in file_vars or sensitive_name(var) or any(secret in value for secret in secrets):
                shown = "<redacted>"
            else:
                shown = json.dumps(value, ensure_ascii=True)
            status = "相同" if value == first.get(var) else "不同"
            print("  %s=%s [%s%s]" % (var, shown, status, ", 已放行" if var in allowed else ""))


def prepare_envs(a, m):
    required, file_vars = read_required_env(a.require_env_file) if a.require_env_file else ({}, set())
    allowed = set(a.allow_diff)
    if allowed - set(required):
        raise ValueError("--allow-diff 必须指定必需环境文件中的变量")
    base = dict(os.environ, PYTHONIOENCODING="utf-8")
    source_env = {k: str(v) for k, v in (m.get("_strata_cfg_data", {}).get("env") or {}).items()}
    server_envs, engine_envs = {}, {}
    source_cfg = m.get("_strata_cfg_data", {})
    for cfg in m["configs"]:
        server_env = dict(base)
        server_env.update(cfg.get("env", {}))
        # 避免把源配置里的密钥复制到输出文件；服务端同样接受环境变量。
        raw_key = source_cfg.get("api_key") if cfg["kind"] == "strata" else None
        if raw_key and "STRATA_API_KEY" not in server_env:
            if isinstance(raw_key, list):
                if any("," in str(key) for key in raw_key):
                    raise ValueError("含逗号的列表密钥不能安全转入环境；请使用环境变量")
                raw_key = ",".join(str(key) for key in raw_key)
            server_env["STRATA_API_KEY"] = str(raw_key)
        engine_env = dict(server_env)
        if cfg["kind"] == "strata":
            engine_env.update(source_env)
        engine_env.update(cfg.get("env", {}))
        server_envs[cfg["name"]], engine_envs[cfg["name"]] = server_env, engine_env
    if a.check:
        env_summary(m["configs"], engine_envs, required, file_vars, allowed, source_env)
    errors = []
    for cfg in m["configs"]:
        env = engine_envs[cfg["name"]]
        server_env = server_envs[cfg["name"]]
        if (cfg["kind"] == "strata" and "STRATA_API_KEY" in source_env
                and env.get("STRATA_API_KEY") != server_env.get("STRATA_API_KEY")):
            errors.append("%s: 源配置中的 STRATA_API_KEY 与服务端环境不同，不能安全复制" % cfg["name"])
        for name, value in required.items():
            if name not in allowed:
                if name not in env:
                    errors.append("%s: %s 缺失" % (cfg["name"], name))
                elif env[name] != value:
                    errors.append("%s: %s 不匹配" % (cfg["name"], name))
                if name == "STRATA_API_KEY" and cfg["kind"] == "strata":
                    server_env = server_envs[cfg["name"]]
                    if name not in server_env:
                        errors.append("%s: 服务端 %s 缺失" % (cfg["name"], name))
                    elif server_env[name] != value:
                        errors.append("%s: 服务端 %s 不匹配" % (cfg["name"], name))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise ValueError("环境复制检查失败；未启动引擎")
    return server_envs


def check_paths(a, m, cells, out):
    for name, _, _ in cells:
        load_prompt(out, name)
    tokenizer = a.tokenizer or m.get("tokenizer") or m.get("_strata_cfg_data", {}).get("tokenizer")
    if not isinstance(tokenizer, str) or not tokenizer:
        raise ValueError("检查需要 --tokenizer 或配置中的 tokenizer 路径")
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        if not (Path(tokenizer) / name).is_file() or not os.access(Path(tokenizer) / name, os.R_OK):
            raise ValueError("分词器文件缺失或不可读：%s" % name)
    kinds = {c["kind"] for c in m["configs"]}
    if "strata" in kinds:
        if not (Path(m["serve"]) / "server.py").is_file():
            raise ValueError("serve/server.py 不存在")
        exe = m.get("strata_exe") or m["_strata_cfg_data"].get("exe")
        if not isinstance(exe, str) or not Path(exe).is_file() or not os.access(exe, os.X_OK):
            raise ValueError("Strata 引擎文件不存在或不可执行")
    if "llama" in kinds:
        if (not Path(m["llama_server"]).is_file() or not os.access(m["llama_server"], os.X_OK)
                or not Path(m["gguf"]).is_file()):
            raise ValueError("llama_server 不存在、不可执行，或 gguf 文件不存在")
    if m.get("cwd") and not Path(m["cwd"]).is_dir():
        raise ValueError("cwd 目录不存在")


def http(url, body=None, timeout=3600, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0])
    except Exception:
        return -1


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop, self.vram, self.rss, self.sysused = pid, False, 0, 0, 0

    def run(self):
        while not self.stop:
            self.vram = max(self.vram, vram_used_mib())
            try:
                p = psutil.Process(self.pid)
                self.rss = max(self.rss, sum(c.memory_info().rss for c in [p] + p.children(recursive=True)))
            except psutil.Error:
                pass
            self.sysused = max(self.sysused, psutil.virtual_memory().used)
            time.sleep(1.0)


class Engine:
    def __init__(self, machine, cfg, env):
        self.m, self.cfg, self.env, self.proc, self.log = machine, cfg, env, None, None
        self.api_key = client_key(env.get("STRATA_API_KEY"), machine.get("_strata_cfg_data"), cfg["kind"])

    def command(self):
        c = self.cfg
        if c["kind"] == "llama":
            return [self.m["llama_server"], "-m", self.m["gguf"], "--port", str(PORT), "--host", "127.0.0.1",
                    "-np", "1", "--no-webui", "--jinja"] + c["args"]
        scfg = copy.deepcopy(self.m["_strata_cfg_data"])
        # serve 会再次应用配置中的 env；让本次对照配置优先，且不把密钥写入新配置。
        if scfg.get("env") and c.get("env"):
            scfg["env"] = {k: v for k, v in scfg["env"].items() if k not in c["env"]}
        scfg.pop("api_key", None)
        if scfg.get("env"):
            scfg["env"].pop("STRATA_API_KEY", None)
        scfg["port"] = PORT
        scfg.pop("vision", None)
        args = [a for a in scfg["args"]]
        for k, v in c.get("override", {}).items():          # replace "--flag value" pairs
            if k in args:
                args[args.index(k) + 1] = v
            else:
                args += [k, v]
        scfg["args"] = args + c["args"]
        if self.m.get("strata_exe"):
            scfg["exe"] = self.m["strata_exe"]
        p = Path(self.out) / ("strata-cfg-%s.json" % c["name"])
        config_text = json.dumps(scfg, indent=1)
        for secret in api_secrets(self.env):
            if secret in config_text or json.dumps(secret)[1:-1] in config_text:
                raise ValueError("源配置中还含有密钥，拒绝写入输出文件")
        p.write_text(config_text)
        return [sys.executable, str(Path(self.m["serve"]) / "server.py"), "--engine", "strata", "--config", str(p),
                "--port", str(PORT)]

    def start(self, out):
        self.out = out = str(Path(out).resolve())
        self.base_vram, self.base_used = vram_used_mib(), psutil.virtual_memory().used
        cmd = self.command()
        self.cmdline = safe_command(cmd, api_secrets(self.env))
        self.log = open(Path(out) / ("server-%s-%d.log" % (self.cfg["name"], int(time.time()))), "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=self.env,
                                     cwd=self.m.get("cwd"))
        self.t0 = time.time()
        self.sampler = Sampler(self.proc.pid)
        self.sampler.start()
        health = "http://127.0.0.1:%d/health" % PORT
        while time.time() - self.t0 < 1500:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited rc=%s (see %s)" % (self.proc.returncode, self.log.name))
            try:
                r = http(health, timeout=5)
                if r.get("status", "ok") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("server not ready in 1500 s")
        self.load_s = time.time() - self.t0

    def stop(self):
        if self.proc:
            self.sampler.stop = True
            try:
                p = psutil.Process(self.proc.pid)
                kids = p.children(recursive=True)
                for c in kids + [p]:
                    try:
                        c.terminate()
                    except psutil.Error:
                        pass
                psutil.wait_procs(kids + [p], timeout=30)
                for c in kids + [p]:
                    try:
                        if c.is_running():
                            c.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            self.log.close()
            self.proc = None
            time.sleep(5)

    def chat(self, prompt, temp, max_tokens=GEN, seed=1):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temp, "top_k": 20, "top_p": 0.95, "seed": seed, "stream": False,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        if temp == 0:
            body.pop("top_p"), body.pop("top_k")
        t = time.time()
        r = http("http://127.0.0.1:%d/v1/chat/completions" % PORT, body, api_key=self.api_key)
        wall = time.time() - t
        tm = r.get("timings") or {}
        u = r.get("usage") or {}
        return {"wall_s": round(wall, 2), "prompt_n": tm.get("prompt_n", u.get("prompt_tokens")),
                "prompt_ms": tm.get("prompt_ms"), "prompt_tps": tm.get("prompt_per_second"),
                "pred_n": tm.get("predicted_n", u.get("completion_tokens")), "pred_tps": tm.get("predicted_per_second"),
                "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
                "text": (r["choices"][0]["message"].get("content") or "")}


def cmd_prompts(a):
    sys.path.insert(0, str(Path(a.repo) / "tools"))
    import strata_tokenizer as ST
    tp = Path(a.tokenizer)
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    corpus = "\n\n".join(Path(f).read_text(encoding="utf-8", errors="replace") for f in a.corpus)
    print("corpus chars", len(corpus))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allids = tok.encode(corpus)
    print("corpus tokens", len(allids))
    for name, n, _ in CELLS:
        if len(allids) < n:
            raise SystemExit("corpus too small for %s" % name)
        (out / ("prompt-%s.txt" % name)).write_text(tok.decode(allids[:n]), encoding="utf-8")
        print(name, n, "tokens")


def load_prompt(out, name):
    return (Path(out) / ("prompt-%s.txt" % name)).read_text(encoding="utf-8")


def cmd_run(a):
    m, cfgs, cells = load_machine(a)
    envs = prepare_envs(a, m)
    out = Path(a.out)
    if a.check:
        check_paths(a, m, cells, out)
        print("检查通过；未启动引擎")
        return
    (out / "raw").mkdir(parents=True, exist_ok=True)
    prompts = {c[0]: load_prompt(out, c[0]) for c in cells}
    secrets = [secret for env in envs.values() for secret in api_secrets(env)]
    for rnd in range(a.start_round, a.start_round + a.rounds):
        order = cfgs if rnd % 2 == 0 else cfgs[::-1]
        for cfg in order:
            eng = Engine(m, cfg, envs[cfg["name"]])
            rec_path = out / "raw" / ("%s.jsonl" % cfg["name"])
            try:
                eng.start(str(out))
                eng.chat("Say hello.", 0.0, 16)                              # warm-up, discarded
                eng.chat(prompts["4k"][:6000] if "4k" in prompts else "hi", 0.0, 32)
                for cname, n, reps in cells:
                    for t in TEMPS:
                        for k in range(reps):
                            nonce = "[bench %s r%d %s t%.1f k%d %d]\n" % (cfg["name"], rnd, cname, t, k, time.time())
                            r = eng.chat(nonce + prompts[cname] + "\n\nSummarize the text above in three sentences.",
                                         t, m.get("gen", GEN))
                            r.update(config=cfg["name"], round=rnd, cell=cname, temp=t, rep=k,
                                     vram_peak_mib=eng.sampler.vram, vram_base_mib=eng.base_vram,
                                     rss_peak_gib=round(eng.sampler.rss / 2**30, 2),
                                     sys_used_delta_gib=round((eng.sampler.sysused - eng.base_used) / 2**30, 2),
                                     load_s=round(eng.load_s, 1), cmd=eng.cmdline if (rnd == a.start_round and k == 0) else None)
                            r["text"] = redact(r["text"][:300], secrets)
                            with open(rec_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(r) + "\n")
                            print(cfg["name"], rnd, cname, t, "pp %s tg %s draft %s/%s ttft %s" % (
                                r["prompt_tps"], r["pred_tps"], r["draft_acc"], r["draft_n"], r["prompt_ms"]), flush=True)
            except Exception as e:
                detail = type(e).__name__ if secrets else repr(e)
                print("ERROR", cfg["name"], rnd, detail, flush=True)
                with open(out / "raw" / "errors.log", "a") as f:
                    f.write("%s r%d %s\n" % (cfg["name"], rnd, detail))
            finally:
                eng.stop()


QPROMPT = "Write a short story about a lighthouse keeper."


def cmd_quality(a):
    m, cfgs, _ = load_machine(a)
    envs = prepare_envs(a, m)
    if a.smoke and len(cfgs) < 2:
        raise ValueError("--smoke 需要至少两个配置")
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    prompts = (QPROMPT, "Write a Python function that parses a CSV file into a list of dicts, with tests.")
    res = {}
    for cfg in cfgs:
        eng = Engine(m, cfg, envs[cfg["name"]])
        try:
            eng.start(str(out))
            eng.chat("Say hello.", 0.0, 16)
            for q in prompts:
                r = eng.chat(q, 0.0, 200)
                if a.smoke and not r["text"]:
                    raise ValueError("%s 返回空答案；无法比较" % cfg["name"])
                res.setdefault(cfg["name"], []).append(r["text"])
        except Exception as e:
            if api_secrets(eng.env):
                raise RuntimeError("%s 执行失败（%s）" % (cfg["name"], type(e).__name__)) from None
            raise
        finally:
            eng.stop()
    secrets = [secret for env in envs.values() for secret in api_secrets(env)]
    stored = {name: [redact(text, secrets) for text in answers] for name, answers in res.items()}
    (out / "raw" / "quality.json").write_text(json.dumps(stored, indent=1), encoding="utf-8")
    if a.smoke:
        different = False
        for i in range(len(prompts)):
            answers = [res[cfg["name"]][i].encode("utf-8") for cfg in cfgs]
            same = all(answer == answers[0] for answer in answers[1:])
            print("提示 %d: %s" % (i + 1, "IDENTICAL" if same else "DIFFERS"))
            different |= not same
        if different:
            raise SystemExit(1)


def cmd_summary(a):
    out = Path(a.out)
    rows = {}
    for f in sorted((out / "raw").glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rows.setdefault((r["config"], r["cell"], r["temp"]), []).append(r)
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    print("| config | ctx | temp | n | prompt tok/s | decode tok/s | TTFT s | draft acc | VRAM peak MiB (delta) | RSS GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (c, cell, t), rs in sorted(rows.items()):
        pp = [r["prompt_tps"] for r in rs if r["prompt_tps"]]
        tg = [r["pred_tps"] for r in rs if r["pred_tps"]]
        tt = [r["prompt_ms"] / 1000 for r in rs if r["prompt_ms"]]
        da = [r["draft_acc"] / r["draft_n"] for r in rs if r.get("draft_n")]
        print("| %s | %s | %.1f | %d | %.1f | %.2f | %.2f | %s | %d (+%d) | %.1f |" % (
            c, cell, t, len(rs), med(pp), med(tg), med(tt), ("%.2f" % med(da)) if da else "-",
            max(r["vram_peak_mib"] for r in rs), max(r["vram_peak_mib"] - r["vram_base_mib"] for r in rs),
            max(r["rss_peak_gib"] for r in rs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prompts", "run", "quality", "summary"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--machine")
    ap.add_argument("--configs")
    ap.add_argument("--cells")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--tokenizer")
    ap.add_argument("--repo")
    ap.add_argument("--corpus", nargs="*")
    ap.add_argument("--require-env-file", help="run/quality 启动前核对必需环境变量")
    ap.add_argument("--allow-diff", action="append", default=[], metavar="NAME", help="放行一个必需变量；可重复")
    ap.add_argument("--check", action="store_true", help="run 只检查配置、环境和文件，不启动引擎")
    ap.add_argument("--smoke", action="store_true", help="quality 逐提示比较各配置的完整答案")
    a = ap.parse_args()
    if a.check and a.mode != "run":
        ap.error("--check 仅用于 run")
    if a.smoke and a.mode != "quality":
        ap.error("--smoke 仅用于 quality")
    if (a.require_env_file or a.allow_diff) and a.mode not in ("run", "quality"):
        ap.error("--require-env-file 和 --allow-diff 仅用于 run 或 quality")
    try:
        {"prompts": cmd_prompts, "run": cmd_run, "quality": cmd_quality, "summary": cmd_summary}[a.mode](a)
    except (OSError, ValueError, UnicodeError, RuntimeError) as e:
        print("错误: %s" % e, file=sys.stderr)
        raise SystemExit(1) from None
