import torch
import numpy as np
# Model
model = torch.hub.load('ultralytics/yolov3', 'yolov3_tiny')  # or 'yolov3_spp', 'yolov3_tiny'

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference

print(model)
image = model(img)
image.show()
w = model.state_dict()
w = list(w.values())
# print(len(w[16]))
# with open(r'/home/nguyentienbao/Documents/PycharmProjects/img_preprocessing/test.txt', 'w') as f:
#     f.write("\n".join(map(str, w[16])))
for i in range(len(w)):
    new_weights = w[i]
    weights = new_weights.cpu().detach().numpy()
    if i % 2 == 0:
        with open(r'/home/bao/Documents/Scripts/tiny_yolov3/weights/weight' + str(i) + '.npy', 'wb') as f:
            np.save(f, weights)
    else:
        with open(r'/home/bao/Documents/Scripts/tiny_yolov3/biases/biases' + str(i-1) + '.txt', 'w') as f:
            f.write("\n".join(map(str, weights)))
            f.close()
results = model(img)
results.print()  # or .show(), .save()
