def compute_hybrid_error(ir_data, cam_err):
    """
    Computes a hybrid error based on IR and Camera data.
    If IR is completely lost, it relies exclusively on the Camera.
    Otherwise, it blends IR and Camera errors.
    """
    ir_err, ir_lost, _ = ir_data
    if ir_lost:
        return cam_err
    else:
        return 0.55 * ir_err + 0.45 * cam_err
