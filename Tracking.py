
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

## Setting device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.float16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def model_setup(sam2_checkpoint, model_cfg):
    global predictor

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def assign_point(video_dir, point_dict):
    global predictor, frame_names

    os.makedirs("annotated_frame", exist_ok=True)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    p_len = len(point_dict[1])
    n_len = len(point_dict[0])

    point_order = point_dict[1] + point_dict[0]
    point_PosNeg = [1] * p_len + [0] * n_len

    # Add points
    # points = np.array([[580, 200],[800,400],[200,600],[900,700],[630,100]], dtype=np.float32)
    points = np.array(point_order, dtype=np.float32)

    # for labels, `1` means positive click and `0` means negative click
    # labels = np.array([1,1,1,1,0], np.int32)
    labels = np.array(point_PosNeg, np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    output_image_path = f"./annotated_frame/output_frame_{ann_frame_idx}.png"
    plt.savefig(output_image_path, bbox_inches="tight")

    print(f"Annotation saved to './annotated_frame/output_frame_{ann_frame_idx}.png'.")

    return inference_state

def seg(output_dir, video_dir, inference_state, vis_frame_stride):
    global frame_names

    os.makedirs(output_dir, exist_ok=True)

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    # vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        output_frame_path = os.path.join(output_dir, f"segmented_frame_{out_frame_idx}.png")
        plt.savefig(output_frame_path, bbox_inches="tight")
        plt.close()

    print(f"Segmented video frame successfully. Saved to folder {output_dir}.")


if __name__ == "__main__":
    sam2_checkpoint = '/home/hc4549/HandTracking/sam2.1_hiera_large.pt' #Weight path (downloaded from official SAM web)
    model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml' #Config path

    video_dir = 'videoframe' #Directory that contian all the frames
    #"Clicks" coordination, should be in this form, 1 means positive click, while 0 means negative click.
    point_dict = {1 : [[580, 200],[800,400],[200,600],[900,700]], 
                  0 : [[630,100]]}

    output_dir = 'segmented_frames' #Output folder

    #Initialize model
    model_setup(sam2_checkpoint, model_cfg)

    #Assign point to first frame
    #This will save the segmented first frame accoridng to the "Clicks", check if it's segmented as desire before cotinue to next part.
    inference_state = assign_point(video_dir, point_dict) 

    #After making sure the first frame is segmented correctly, cotinue below function, it will output all the segmented frames in the output_dir.
    seg(output_dir, video_dir, inference_state, vis_frame_stride=1) #vis_frame means how many frame user want to output, 1=output every frame, 30=output one every 30 frames.
