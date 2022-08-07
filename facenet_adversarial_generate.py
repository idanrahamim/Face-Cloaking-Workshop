"""
This module is responsible to mange the generation of adversarial examples.
It contains the functions that used to conduct the research presented in the report.
It also contains a function named 'execute_attack' that is responsible to generate adversarial examples
for the website.
The function 'simple_run' is used to demonstrate the stages in generating adversarial examples.

"""

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

working_directory = './MYWORKINGDIRS'
if os.path.exists(working_directory):
    shutil.rmtree(working_directory)
os.mkdir(working_directory)

face_detector = MTCNN(image_size=240, margin=0, min_face_size=20)
face_embedding = InceptionResnetV1(pretrained='vggface2').eval()
face_embedding_vgg = InceptionResnetV1(pretrained='vggface2').eval()
face_embedding_casia = InceptionResnetV1(pretrained='casia-webface').eval()

margin = 10.


def _l2(x1, x2, the_original_input=None, the_pertubation=None):
    return ((x1 - x2) ** 2).sum()


def _faceoff_loss(x_new, x_orig, the_original_input=None, the_pertubation=None, face_embedding=None,
                  with_pertubation_loss: bool = True):
    """
    This is our implementation for faceoff loss.
    We added a perturbation loss that as explained in the report improves the results.
    :param x_new: the embedding of the adversarial image.
    :param x_orig: the embedding of the original image.
    :param the_original_input: the original image (not embedding, actual representation).
                               not in use here: put for convention with ulixes loss
    :param the_pertubation: the adversarial image - original image
    :param face_embedding: a face embedding network, not in use here: put for convention with ulixes loss.
    :param with_pertubation_loss: True to use the permutation loss. False for the original face-off implementation.
    """
    if x_new.equal(x_orig):
        x_orig = x_orig + 1e-8 * torch.ones_like(x_orig)
    l2_loss = -_l2(x_new, x_orig)
    if with_pertubation_loss:
        p_loss = _pertubation_loss_1(the_pertubation) / 15
        return l2_loss + 0.25 * p_loss
    else:
        return l2_loss


def _ulixes_loss(x_new, x_orig, the_original_input=None, the_pertubation=None, face_embedding=None):
    """
    This is our implementation for ulixes loss.
    :param x_new: the embedding of the adversarial image.
    :param x_orig: the embedding of the original image.
    :param the_original_input: the original image (not embedding, actual representation).
    :param the_pertubation: the adversarial image - original image
    :param face_embedding: a face embedding network.
    """
    epsilon = 1e-8 * torch.rand_like(the_original_input)
    n = face_embedding(the_original_input + epsilon)
    p = x_orig
    a = x_new
    loss = torch.relu(_l2(a, n) - _l2(a, p) + margin)
    print(loss)
    return loss + 0.01 * _pertubation_loss_1(the_pertubation)


def _pertubation_loss_1(p):
    """ 'punishes' high perturbation values much more than small perturbation values """
    return (p ** 2).sum()


def _evaluate_algorithm(input_images: datasets.ImageFolder,
                        fr_system: face_recognition_system.FaceRecognitionSystem):
    success_counter = 0
    idx_to_class = {v: k for k, v in input_images.class_to_idx.items()}
    for batch_num, (xs, ys) in enumerate(input_images):
        real_name = idx_to_class[ys]
        recognized_as = fr_system.recognize_by_image(xs)['name']
        if real_name != recognized_as:
            success_counter += 1

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
                attack_loss = _ulixes_loss if attack_loss_name == 'U' else _faceoff_loss
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
                                                     parameters={'epsilon': epsilon, 'steps': iterations,
                                                                 'alpha': alpha},
                                                     face_detector=face_detector, amplification=amplification)
                                attack.apply_attack(users_images,
                                                    device=device)
                                e = time.time()

                                cloaked_images = datasets.ImageFolder(output_directory)

                                acc_fake = 1. - _evaluate_algorithm(cloaked_images, face_recognition_sys)
                                acc_real = 1. - _evaluate_algorithm(users_images, face_recognition_sys)
                                out_report_file.write(
                                    f"\n{gym_size},{attack_loss_name},{wb},{amplification},{iterations},{epsilon},"
                                    f"{alpha},{acc_fake},{acc_real},{e - s}")
                                out_report_file.flush()

                                # shutil.copytree('./out/Abradley_cooper', f'./OUT_FOR_RESULTS/Abradley_cooper_{a}')
                                # # os.rename('./OUT_FOR_RESULTS/Abradley_cooper', f'./OUT_FOR_RESULTS/')
                                # shutil.copytree('./out/Nick_Frost', f'./OUT_FOR_RESULTS/Nick_Frost_{a}')
                                # # os.rename('./OUT_FOR_RESULTS/Nick_Frost', f'./OUT_FOR_RESULTS/Nick_Frost_{a}')


def simple_run(face_embedding_net=face_embedding_vgg):
    """
    :param face_embedding_net: For black-box execution change to face_embedding_casia.
                               (The default value is face_embedding_vgg which is for white-box).

    This is the 'main' function for this module.
    To execute this function make sure you have the folders:
        './GYM_MEMBERS2' - the users for the face recognition system.
        './GYM_MEMBERS3' - small set of users to attack. subset of './GYM_MEMBERS2'.

    As GYM_MEMBERS2 is bigger the results are better.
    The function prints the face recognition success rate on fake and real images.
    """
    # pertubation generation
    users_images = datasets.ImageFolder('./GYM_MEMBERS3')
    attack = attacks.PGD(face_embedding_net, _ulixes_loss,
                         parameters={'epsilon': 0.05, 'steps': 20, 'alpha': 0.005},
                         face_detector=face_detector, amplification=1.)
    attack.apply_attack(users_images,
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
          f"{1. - _evaluate_algorithm(cloaked_images, face_recognition_sys)}")
    print(f"FACE RECOGNITION SUCCESS RATE ON REAL IMAGES IS: "
          f"{1. - _evaluate_algorithm(users_images, face_recognition_sys)}")


def execute_attack(input_image, out_dir, first_amp=4.2, second_amp=6.):
    """
    This function is responsilbe to generate adversarial images for the use of the website
    :param input_image: the input image path
    :param out_dir: the output directory path
    :param first_amp: first image amplification factor
    :param second_amp: second image amplification factor
    :return: 3 paths, original_image, attack 1 output image, attack 2 output image
    """
    random_str = os.urandom(7).hex()
    process_dir = 'MYWORKINGDIR_' + random_str
    process_working_dir = os.path.join(working_directory, process_dir)
    if os.path.exists(process_working_dir):
        shutil.rmtree(process_working_dir)
    os.mkdir(process_working_dir)
    input_work_dir = os.path.join(process_working_dir, 'INPUT_IMAGE')
    os.mkdir(input_work_dir)
    output_mid = os.path.join(input_work_dir, 'out')
    os.mkdir(output_mid)
    shutil.copy2(input_image, input_work_dir)

    # The first attack
    users_images = datasets.ImageFolder(process_working_dir)
    attack = attacks.PGD(face_embedding_vgg, _ulixes_loss,
                         parameters={'epsilon': 0.03, 'steps': 10, 'alpha': 0.005},
                         face_detector=face_detector, amplification=first_amp)
    attack.apply_attack(users_images,
                        device=device,
                        output_dir=output_mid)
    new_path = os.path.join(output_mid, 'first_' + os.path.basename(input_image))
    os.rename(os.path.join(output_mid, 'INPUT_IMAGE', '0.png'), new_path)
    shutil.copy2(new_path, out_dir)

    # The second attack
    shutil.rmtree(output_mid)
    os.mkdir(output_mid)

    attack = attacks.PGD(face_embedding_vgg, _ulixes_loss,
                         parameters={'epsilon': 0.03, 'steps': 10, 'alpha': 0.005},
                         face_detector=face_detector, amplification=second_amp)

    attack.apply_attack(users_images,
                        device=device,
                        output_dir=output_mid)

    new_ulixes_path = os.path.join(output_mid, 'second_' + os.path.basename(input_image))
    os.rename(os.path.join(output_mid, 'INPUT_IMAGE', '0.png'), new_ulixes_path)
    shutil.copy2(new_ulixes_path, out_dir)

    to_original_path = os.path.join(out_dir, 'original_' + os.path.basename(input_image))
    if os.path.exists(to_original_path):
        os.remove(to_original_path)
    os.rename(input_image, to_original_path)

    return os.path.join(out_dir, os.path.basename(new_path).replace('first', 'original')), \
        os.path.join(out_dir, os.path.basename(new_path)), \
        os.path.join(out_dir, os.path.basename(new_ulixes_path))
