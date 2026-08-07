import base64
from jupyter_client.manager import KernelManager

class KernelSession:
    """Wraps a single persistent Jupyter kernel: start once, execute many
    times, state (variables, imports) carries over between calls."""
    
    def __init__(self):
        print("Creating KernelManager...")
        self.km = KernelManager(kernel_name="python3")

        print("Starting kernel...")
        self.km.start_kernel()

        print("Creating client...")
        self.kc = self.km.client()

        print("Starting channels...")
        self.kc.start_channels()

        print("Waiting for ready...")
        self.kc.wait_for_ready(timeout=30)

        print("Kernel ready!")
        
    
    def execute(self, code: str, timeout: int = 20) -> dict:
        """Run code in the kernel. Returns a dict with stdout, a text
        result, any image (base64 PNG), and an error traceback if one
        occurred. Exactly one of result/error/images will be meaningful."""
        
        msg_id = self.kc.execute(code)
        stdout = ""
        result_text = None
        error_text = None
        images = []
        
        while(True):
            msg = self.kc.get_iopub_msg(timeout=timeout)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            
            msg_type = msg["msg_type"]
            content = msg["content"]
            
            if msg_type == "stream":
                stdout += content["text"]
            elif msg_type in ("execute_result","display_data"):
                data = content.get("data", {})
                
                if "text/plain" in data:
                    result_text = data["text/plain"]
                
                if "image/png" in data:
                    images.append(data["image/png"])
                    
            elif msg_type == "error":
                error_text = "\n".join(content["traceback"])
                
            elif msg_type == "status" and content["execution_state"] == "idle":
                break
            
            
        while True:
            reply = self.kc.get_shell_msg(timeout=timeout)
            if reply["parent_header"].get("msg_id") == msg_id:
                break
        
        return {
            "stdout": stdout,
            "result": result_text,
            "error": error_text,
            "images": images,
        }
        
    def shutdown(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)
        
