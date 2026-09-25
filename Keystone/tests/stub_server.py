"""A local OpenAI-compatible chat-completions server for tests: a stub judge and a stub teacher."""
import json
import re
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Stub:
    def __init__(self):
        self.requests = []          # (model, last user message)
        self.lock = threading.Lock()
        self.fail_next = 0          # respond 500 to the next N requests
        self.garbled_next = 0       # answer with an unparseable verdict N times
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()

    def overrides(self):
        return [f"judge.base_url={self.base_url}", f"teachers.offpolicy.base_url={self.base_url}"]

    def _handler(self):
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                msg = body["messages"][-1]["content"]
                with stub.lock:
                    stub.requests.append((body["model"], msg))
                    if stub.fail_next > 0:
                        stub.fail_next -= 1
                        self.send_response(500); self.end_headers(); return
                    garble = stub.garbled_next > 0
                    if garble:
                        stub.garbled_next -= 1
                text = "I cannot tell." if garble else stub.answer(body["model"], msg)
                out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": text}}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        return H

    @staticmethod
    def answer(model, msg):
        h = zlib.crc32(msg.encode())
        if model == "stub-teacher":
            m = re.search(r"(\d+)\+1\?", msg)
            if m:
                return f"<think>add one</think>\\boxed{{{int(m.group(1)) + 1}}}"
            return "Sure thing."
        if "VERDICT: <compliance" in msg:
            return "Reasoning.\nVERDICT: " + ["compliance", "refusal", "partial_refusal"][h % 3]
        if "VERDICT: A" in msg:
            return "Reasoning.\nVERDICT: " + ["A", "B", "TIE"][h % 3]
        if "SCORE:" in msg:
            return f"Reasoning.\nSCORE: {h % 10 + 1}"
        return "ok"
