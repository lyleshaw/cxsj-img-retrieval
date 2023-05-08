import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms


def extract_feature_query(model: torch.nn.Module, img, use_gpu=True) -> torch.tensor:
    c, h, w = img.size()
    img = img.view(-1, c, h, w)
    use_gpu = use_gpu and torch.cuda.is_available()
    img = img.cuda() if use_gpu else img
    input_img = Variable(img)
    outputs = model(input_img)
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff


def load_query_image(query_path: str):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    query_image = datasets.folder.default_loader(query_path)
    query_image = data_transforms(query_image)
    return query_image


def load_model(pretrained_model: str = None, use_gpu: bool = True) -> torch.nn.Module:
    """
    :param use_gpu:
    :param pretrained_model:
    :return:
    """
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  # number of training classes
    model.fc = nn.Sequential(*add_block)
    # model.load_state_dict(torch.load(pretrained_model))

    # remove the final fc layer
    model.fc = nn.Sequential()
    # change to test modal
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model


# sort the images
def sort_img(qf, gf):
    score = gf * qf
    score = score.sum(1)
    print(score)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    import numpy as np
    s = np.around(s, 3)
    return s, index


def add_img(model: torch.nn.Module, img_path):
    gallery = np.load("gallery.npy", allow_pickle=True)
    path_list = np.load("path.npy", allow_pickle=True).tolist()
    print(path_list)
    features = torch.tensor(gallery)
    query_image = load_query_image(img_path)
    feature = extract_feature_query(model=model, img=query_image)
    features = torch.cat((features, feature), 0)
    path_list += [img_path]
    print(path_list)
    np.save("gallery.npy", features)
    np.save("path.npy", np.array(path_list))


def query_img(model: torch.nn.Module, img_path):
    gallery_feature = np.load("gallery.npy")
    path_list = np.load("path.npy").tolist()

    query_image = load_query_image(img_path)
    query_feature = extract_feature_query(model=model, img=query_image)

    similarity, index = sort_img(torch.tensor(query_feature), torch.tensor(gallery_feature))
    sorted_paths = [path_list[i] for i in index]

    print(sorted_paths)

    return sorted_paths


if __name__ == '__main__':
    model = load_model(use_gpu=True)
    add_img(model, 'data/query/1.jpg')
    query_img(model, 'data/query/query.jpg')
