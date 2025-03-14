import importlib

from mug.util import get_obj_from_str

def instantiate_from_config(config):
    if isinstance(config, dict):
        if "class_path" in config:
            # Lightning CLI style config
            module_path, class_name = config["class_path"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(**(config.get("init_args", {})))
        elif "target" in config:
            # Old style config
            return get_obj_from_str(config["target"])(**config.get("params", {}))
    return config