import numpy as np

def state_metrics(truth,pred,lanes,mask):
    if not mask.any(): raise ValueError('No validation targets')
    speed=float(np.sqrt(np.mean((truth[...,0][mask]-pred[...,0][mask])**2)))
    per_lane=(truth[...,1]-pred[...,1])/np.asarray(lanes)[None]
    flow=float(np.sqrt(np.mean(per_lane[mask]**2)))
    return {'rmse_speed_kmh':speed,'rmse_flow_vph_per_lane':flow,
            'S_state':.54*max(0,1-speed/25)+.46*max(0,1-flow/600),'n_targets':int(mask.sum())}

def proxy_queue_iou(truth,pred,valid):
    intersection=np.sum(truth & pred & valid); union=np.sum((truth|pred)&valid)
    return float(intersection/union) if union else 1.
