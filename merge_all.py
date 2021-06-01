import h5py
import numpy as np
import os, glob, shutil

SRC_DIR = "./processed/"
DST_DIR = "./merged/"

TRAIN_SUBJECT = ["S1", "S5", "S6", "S7", "S8"]
TEST_SUBJECT = ["S9", "S11"]

CATEGORY = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Purchases", "Sitting", "SittingDown", "Smoking", "TakingPhoto","Waiting","Walking", "WalkingDog", "WalkingTogether"]
CAMERA_ID = ["54138969","55011271","58860488","60457274"]
NUMBER = ["-1","-2"]

os.makedirs(os.path.join('./merged'), exist_ok=True)
os.makedirs(os.path.join(DST_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(DST_DIR, 'test'), exist_ok=True)

def merge_trainset():
    merged_2d = None
    merged_3d = None
    merged_angle = None
    merged_bbox = None
    merged_action = None
    merged_subject_id = None
    file_id = 0
    for subject in TRAIN_SUBJECT:
        print(subject)
        for category in CATEGORY:
            for num in NUMBER:
                data = h5py.File(os.path.join(SRC_DIR, subject, category+num,"annot.h5"), 'r')
                pose_2d = data["pose"]['2d']
                pose_3d = data["pose"]['3d']
                angle = data["pose"]['angle']
                bbox = data["pose"]['bbox']
                action = data['action']
                subject_id = data['subject'] 
                img_path = []
                for id in CAMERA_ID:
                    img_path += sorted(glob.glob(os.path.join(SRC_DIR, subject, category+num,"imageSequence", id, "*.jpg")))

                assert(pose_2d.shape[0] == pose_3d.shape[0] == angle.shape[0] == bbox.shape[0] == action.shape[0] == subject_id.shape[0] == len(img_path))

                if merged_2d is None:
                    merged_2d = pose_2d
                    merged_3d = pose_3d
                    merged_angle = angle
                    merged_bbox = bbox
                    merged_action = action
                    merged_subject_id = subject_id
                else:
                    merged_2d = np.vstack((merged_2d, pose_2d))
                    merged_3d = np.vstack((merged_3d, pose_3d))
                    merged_angle = np.vstack((merged_angle, angle))
                    merged_bbox = np.vstack((merged_bbox, bbox))
                    merged_action = np.hstack((merged_action, action))
                    merged_subject_id = np.hstack((merged_subject_id, subject_id))
                
                for i in range(len(img_path)):
                    shutil.copy2(img_path[i], os.path.join(DST_DIR, "train", str(file_id).zfill(6) + ".jpg"))
                    file_id += 1
                    
    assert(merged_2d.shape[0] == merged_3d.shape[0] == merged_action.shape[0] == merged_angle.shape[0] == merged_bbox.shape[0] == merged_subject_id.shape[0] == file_id)
    np.savez(os.path.join(DST_DIR, 'train', 'annot.npz'), pose_2d=merged_2d, pose_3d=merged_3d, angle=merged_angle, action=merged_action, bbox=merged_bbox, subject=merged_subject_id)

    print("Done")

def merge_testset():
    merged_2d = None
    merged_3d = None
    merged_angle = None
    merged_bbox = None
    merged_action = None
    merged_subject_id = None
    file_id = 0
    for subject in TEST_SUBJECT:
        print(subject)
        for category in CATEGORY:
            for num in NUMBER:
                data = h5py.File(os.path.join(SRC_DIR, subject, category+num,"annot.h5"), 'r')
                pose_2d = data["pose"]['2d']
                pose_3d = data["pose"]['3d']
                angle = data["pose"]['angle']
                bbox = data["pose"]['bbox']
                action = data['action']
                subject_id = data['subject']
                img_path = []
                for id in CAMERA_ID:
                    img_path.extend(sorted(glob.glob(os.path.join(SRC_DIR, subject, category+num,"imageSequence", id, "*.jpg"))))
                
                assert(pose_2d.shape[0] == pose_3d.shape[0] == angle.shape[0] == bbox.shape[0] == action.shape[0] == subject_id.shape[0] == len(img_path))

                if merged_2d is None:
                    merged_2d = pose_2d
                    merged_3d = pose_3d
                    merged_angle = angle
                    merged_bbox = bbox
                    merged_action = action
                    merged_subject_id = subject_id
                else:
                    merged_2d = np.vstack((merged_2d, pose_2d))
                    merged_3d = np.vstack((merged_3d, pose_3d))
                    merged_angle = np.vstack((merged_angle, angle))
                    merged_bbox = np.vstack((merged_bbox, bbox))
                    merged_action = np.hstack((merged_action, action))
                    merged_subject_id = np.hstack((merged_subject_id, subject_id))
                
                for i in range(len(img_path)):
                    shutil.copy2(img_path[i], os.path.join(DST_DIR, "test", str(file_id).zfill(6) + ".jpg"))
                    file_id += 1
                    
    assert(merged_2d.shape[0] == merged_3d.shape[0] == merged_action.shape[0] == merged_angle.shape[0] == merged_bbox.shape[0] == merged_subject_id.shape[0] == file_id)
    np.savez(os.path.join(DST_DIR, 'test', 'annot.npz'), pose_2d=merged_2d, pose_3d=merged_3d, angle=merged_angle, action=merged_action, bbox=merged_bbox, subject=merged_subject_id)
    print("Done")

if __name__ == '__main__':
  merge_trainset()
  merge_testset()
