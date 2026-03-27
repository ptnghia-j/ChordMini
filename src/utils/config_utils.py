from __future__ import annotations


def get_config_value(config, section, key, default):
    if hasattr(config, section) and not callable(getattr(config, section, None)):
        sub = getattr(config, section)
        if isinstance(sub, dict):
            return sub.get(key, default)
        return getattr(sub, key, default)
    if hasattr(config, 'get'):
        return config.get(section, {}).get(key, default)
    return default


__all__ = ['get_config_value']
