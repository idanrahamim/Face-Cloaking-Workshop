import time
import attacks
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
import os
import shutil
import face_recognition_system

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_directory = './out'
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.mkdir(output_directory)

face_detector = MTCNN(image_size=240, margin=0, min_face_size=20)
face_embedding = InceptionResnetV1(pretrained='vggface2').eval()
face_embedding_vgg = InceptionResnetV1(pretrained='vggface2').eval()
face_embedding_casia = InceptionResnetV1(pretrained='casia-webface').eval()

margin = 10.


def face_distinguish_loss_function(x_new, x_orig, the_original_input=None, the_pertubation=None, face_embedding=None):
    if x_new.equal(x_orig):
        x_orig = x_orig + 1e-8 * torch.ones_like(x_orig)
    l2_loss = -l2(x_new, x_orig)
    p_loss = pertubation_loss_1(the_pertubation) / 15
    print(l2_loss)
    return l2_loss + 0.25 * p_loss


def l2(x1, x2, the_original_input=None, the_pertubation=None):
    return ((x1 - x2) ** 2).sum()


def ulixes_loss(x_new, x_orig, the_original_input=None, the_pertubation=None, face_embedding=None):
    epsilon = 1e-8 * torch.rand_like(the_original_input)
    n = face_embedding(the_original_input + epsilon)
    p = x_orig
    a = x_new
    loss = torch.relu(l2(a, n) - l2(a, p) + margin)
    print(loss)
    return loss + 0.01 * pertubation_loss_1(the_pertubation)


def pertubation_loss_1(p):
    return (p ** 2).sum()


def execute_attack(input_image, out_dir):
    if os.path.exists('./MYWORKINGDIR'):
        shutil.rmtree('./MYWORKINGDIR')
    os.mkdir('./MYWORKINGDIR')
    os.mkdir('./MYWORKINGDIR/INPUT_IMAGE')
    output_mid = './MYWORKINGDIR/INPUT_IMAGE/out'
    os.mkdir(output_mid)
    shutil.copy2(input_image, './MYWORKINGDIR/INPUT_IMAGE')

    # 50% correct recognition rate

    users_images = datasets.ImageFolder('./MYWORKINGDIR')
    attack = attacks.PGD(face_embedding_vgg, ulixes_loss,
                         hp={'epsilon': 0.03, 'steps': 10, 'alpha': 0.005},
                         face_detector=face_detector, amplification=4.2)
    attack.test_attack(users_images,
                       device=device,
                       output_dir=output_mid)
    new_path = os.path.join(output_mid, '50_' + os.path.basename(input_image))
    os.rename(os.path.join(output_mid, 'INPUT_IMAGE', '0.png'), new_path)
    shutil.copy2(new_path, out_dir)

    # 20% correct recognition rate

    shutil.rmtree(output_mid)
    os.mkdir(output_mid)

    attack = attacks.PGD(face_embedding_vgg, ulixes_loss,
                         hp={'epsilon': 0.03, 'steps': 10, 'alpha': 0.005},
                         face_detector=face_detector, amplification=6.)

    attack.test_attack(users_images,
                       device=device,
                       output_dir=output_mid)

    new_ulixes_path = os.path.join(output_mid, '20_' + os.path.basename(input_image))
    os.rename(os.path.join(output_mid, 'INPUT_IMAGE', '0.png'), new_ulixes_path)
    shutil.copy2(new_ulixes_path, out_dir)

    to_original_path = os.path.join(out_dir, 'original_' + os.path.basename(input_image))
    if os.path.exists(to_original_path):
        os.remove(to_original_path)
    os.rename(input_image, to_original_path)

    return os.path.join(out_dir, os.path.basename(new_path).replace('50', 'original')), \
           os.path.join(out_dir, os.path.basename(new_path)), \
           os.path.join(out_dir, os.path.basename(new_ulixes_path))


def evaluate_algorithm(input_images: datasets.ImageFolder,
                       fr_system: face_recognition_system.FaceRecognitionSystem):
    success_counter = 0
    idx_to_class = {v: k for k, v in input_images.class_to_idx.items()}
    for batch_num, (xs, ys) in enumerate(input_images):
        real_name = idx_to_class[ys]
        recognized_as = fr_system.recognize_by_image(xs)['name']
        if real_name != recognized_as:
            success_counter += 1

    return success_counter / len(input_images)


def gpm_evaluate_algorithm(input_images: datasets.ImageFolder,
                           fr_system: face_recognition_system.FaceRecognitionSystem):
    success_counter = 0
    idx_to_class = {v: k for k, v in input_images.class_to_idx.items()}
    for batch_num, (xs, ys) in enumerate(input_images):
        real_name = idx_to_class[ys]
        recognized_as = fr_system.recognize_by_image(xs, ret_order_of_users=True)
        success_counter += recognized_as.index(real_name) / (len(recognized_as) - 1)

    return success_counter / len(input_images)


def parameters_experiment():
    """
        this function is for parameter experiment where we test performance of many different combinations.
        This function is used to generate the results we present in the report.
    """

    users_images = datasets.ImageFolder('./IN')
    with open('./run_report.txt', 'w') as out_report_file:
        out_report_file.write(
            "gym_size,attack_loss_name,whitebox_blackbox,amplification,iterations,epsilon,alpha,acc_fake,acc_real,duration")
        out_report_file.flush()

        for gym_size in [500]:
            face_recognition_users_images_folder = f'./GYM_{gym_size}'
            face_recognition_sys = face_recognition_system.create_face_recognition_system(
                users_images_folder=face_recognition_users_images_folder,
                pretrained_on='vggface2'
            )

            for attack_loss_name in ['F']:
                attack_loss = ulixes_loss if attack_loss_name == 'U' else face_distinguish_loss_function
                for wb in ['B']:
                    face_embedding = face_embedding_casia if wb == 'B' else face_embedding_vgg
                    for amplification in [1.5, 3., 4.5, 6., 8.]:
                        # for iterations in [5, 10, 20, 30, 40, 60, 80, 100]:
                        for iterations in [20]:
                            for epsilon in [0.03, 0.05, 0.08]:
                                alpha = 2. * epsilon / iterations
                                a = f"{gym_size},{attack_loss_name},{wb},{amplification},{iterations},{epsilon},{alpha}"
                                print(a)

                                if os.path.exists(output_directory):
                                    shutil.rmtree(output_directory)
                                os.mkdir(output_directory)

                                s = time.time()
                                attack = attacks.PGD(face_embedding, attack_loss,
                                                     hp={'epsilon': epsilon, 'steps': iterations, 'alpha': alpha},
                                                     face_detector=face_detector, amplification=amplification)
                                attack.test_attack(users_images,
                                                   device=device)
                                e = time.time()

                                cloaked_images = datasets.ImageFolder(output_directory)

                                acc_fake = 1. - evaluate_algorithm(cloaked_images, face_recognition_sys)
                                acc_real = 1. - evaluate_algorithm(users_images, face_recognition_sys)
                                out_report_file.write(
                                    f"\n{gym_size},{attack_loss_name},{wb},{amplification},{iterations},{epsilon},{alpha},{acc_fake},{acc_real},{e - s}")
                                out_report_file.flush()

                                # shutil.copytree('./out/Abradley_cooper', f'./OUT_FOR_RESULTS/Abradley_cooper_{a}')
                                # # os.rename('./OUT_FOR_RESULTS/Abradley_cooper', f'./OUT_FOR_RESULTS/')
                                # shutil.copytree('./out/Nick_Frost', f'./OUT_FOR_RESULTS/Nick_Frost_{a}')
                                # # os.rename('./OUT_FOR_RESULTS/Nick_Frost', f'./OUT_FOR_RESULTS/Nick_Frost_{a}')


def simple_run():
    # pertubation generation
    users_images = datasets.ImageFolder('./GYM_MEMBERS3')
    attack = attacks.PGD(face_embedding_vgg, ulixes_loss,
                         hp={'epsilon': 0.05, 'steps': 20, 'alpha': 0.005},
                         face_detector=face_detector, amplification=1.)
    attack.test_attack(users_images,
                       device=device)

    face_recognition_users_images_folder = './GYM_MEMBERS2'

    # validate that all input users are in the face recognition users.
    face_recognition_users_images_folder_names = os.listdir(face_recognition_users_images_folder)
    for fn in os.listdir('./GYM_MEMBERS3'):
        if fn not in face_recognition_users_images_folder_names:
            raise Exception("input user (for perturbation) is not in the face recognition users")

    # initialize the face recognition system - this part takes massive amount lot of time.
    # decrease the size of the GYM_MEMBERS folder to reduce it.
    face_recognition_sys = face_recognition_system.create_face_recognition_system(
        users_images_folder=face_recognition_users_images_folder,
        pretrained_on='vggface2'
    )
    cloaked_images = datasets.ImageFolder(output_directory)

    print(f"FACE RECOGNITION SUCCESS RATE ON FAKE IMAGES IS: "
          f"{1. - evaluate_algorithm(cloaked_images, face_recognition_sys)}")
    print(f"FACE RECOGNITION SUCCESS RATE ON REAL IMAGES IS: "
          f"{1. - evaluate_algorithm(users_images, face_recognition_sys)}")


if __name__ == '__main__':
    simple_run()
