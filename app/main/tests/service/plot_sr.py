import os
import cv2
import matplotlib.pyplot as plt

original_images_dir = os.path.abspath(r"D:\Mestrado\ppgca_ia_consumer\app\main\tests\data\cropped_images")
super_res_images_dir = os.path.abspath(r"D:\Mestrado\ppgca_ia_consumer\app\main\tests\data\results_cropped_images")

original_images = sorted([f for f in os.listdir(original_images_dir) if os.path.isfile(os.path.join(original_images_dir, f))])
super_res_images = sorted([f for f in os.listdir(super_res_images_dir) if os.path.isfile(os.path.join(super_res_images_dir, f))])

if len(original_images) != len(super_res_images):
    print("As pastas não contêm o mesmo número de imagens.")
else:
    for original_img, super_res_img in zip(original_images, super_res_images):
        original_img_path = os.path.join(original_images_dir, original_img)
        super_res_img_path = os.path.join(super_res_images_dir, super_res_img)
        
        original = cv2.imread(original_img_path)
        super_res = cv2.imread(super_res_img_path)

        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        super_res = cv2.cvtColor(super_res, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('Imagem Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(super_res)
        plt.title('Super-Resolution')
        plt.axis('off')
        
        plt.show()

# if len(original_images) != len(super_res_images):
#     print("As listas de imagens não contêm o mesmo número de arquivos.")
# else:
#     plt.figure(figsize=(15, 10))  # Configurando o tamanho da figura

#     for i, (original_img, super_res_img) in enumerate(zip(original_images, super_res_images)):
#         original_img_path = os.path.join(original_images_dir, original_img)
#         super_res_img_path = os.path.join(super_res_images_dir, super_res_img)

#         original = cv2.imread(original_img_path)
#         super_res = cv2.imread(super_res_img_path)

#         original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#         super_res = cv2.cvtColor(super_res, cv2.COLOR_BGR2RGB)

#         plt.subplot(2, 3, i+1)
#         plt.imshow(original)
#         plt.title(f'Imagem Original {i+1}')
#         plt.axis('off')

#         plt.subplot(2, 3, i+4)
#         plt.imshow(super_res)
#         plt.title(f'Super-Resolution {i+1}')
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()