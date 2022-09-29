def get_task_interface(task, cfg):
    if task == 'SS':
        from .SS_interface import SS_interface
        return SS_interface(cfg)
    elif task == 'depth':
        from .depth_interface import depth_interface
        return depth_interface(cfg)
    elif task == 'od':
        from .OD_interface import OD_interface
        return OD_interface(cfg)
    elif task == '3dod':
        from .stereo_3dod_interface import stereo3dod_interface
        return stereo3dod_interface(cfg)


