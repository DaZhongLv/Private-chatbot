
import os
from apps.simulation_copilot.app import demo

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, show_error=True)

