import traceback
import json
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

try:
    with open("data/extremism_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    item = dataset[0]
    model_kwargs = {"output_attentions": True, "output_hidden_states": True, "attn_implementation": "eager"}
    config = GSPConfig(model_name="meta-llama/Llama-3.2-3B-Instruct", device="cpu", model_kwargs=model_kwargs)
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model("meta-llama/Llama-3.2-3B-Instruct")
        analysis = framework.analyze_text("Test simple text", save_results=False)
        print("Success:", list(analysis.keys()))
except Exception as e:
    traceback.print_exc()
