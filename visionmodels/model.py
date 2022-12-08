from health_multimodal.image import ImageModel

def get_image_model(joint_feature_size=128):
    model = ImageModel("resnet50", joint_feature_size=joint_feature_size)
    del model.encoder.encoder.fc
    return model