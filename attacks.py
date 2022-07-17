import os
import torch
import torchvision.transforms as T

transform = T.ToPILImage()


class Attack:
    def __init__(self, net, loss_fn, face_detector=None, amplification=2.):
        """
        :param net: the network to attack
        :param loss_fn: the attack is with respect to this loss
        """
        self.net = net
        self.loss_fn = loss_fn
        self.name = None
        self.face_detector = face_detector
        self.amplification = amplification

    def perturb(self, X, y, device=None):
        """ generates adversarial examples to inputs (X, y) """
        pass

    def test_attack(self, dataloader,
                    device=None, amplification=2., output_dir='./out'):
        """
        the attack score of attack method A on network <net> is E[A(x) != y] over distribution D when A(x) is the
        constructed adversarial example of attack A on x. We are going to estimate it using samples from test_dataset.

        :param plot_results: plot original vs adv imgs with additional information
        :return: the accuracy on constructed adversarial examples (i.e. 1 - attack score).
        """
        idx_to_class = {v: k for k, v in dataloader.class_to_idx.items()}

        # calculate attack score
        self.net = self.net.to(device)  # move network to device
        for batch_num, (xs, ys) in enumerate(dataloader):
            image_class = idx_to_class[ys]
            orig_image = xs
            boxes, ps = self.face_detector.detect(xs)
            self.face_detector.keep_all = True
            xss = self.face_detector(xs)
            for i, (box, p, xs) in enumerate(zip(boxes, ps, xss)):
                xs = xs.unsqueeze(0)
                # xs = self.face_detector(xs).unsqueeze(0)
                if isinstance(ys, int):
                    ys = torch.tensor([ys])
                xs, ys = xs.to(device), ys.to(device)  # move data to device
                with torch.no_grad():
                    ys = self.net(xs.clone())

                constructed_examples = self.perturb(xs, ys, device=device)
                delta = constructed_examples - xs
                constructed_examples = xs + self.amplification * delta

                width = int(box[2] - box[0])
                height = int(box[3] - box[1])
                ce = constructed_examples[0]
                ce = (ce + 127.5 / 128) / 2

                face = transform(ce).resize((width, height))
                orig_image.paste(face, box=(int(box[0]), int(box[1])))

            os.mkdir(f'{output_dir}/{image_class}')
            orig_image.save(f'{output_dir}/{image_class}/{batch_num}.png')


class PGD(Attack):
    def __init__(self, net, loss_fn, hp, face_detector=None, amplification=2.):
        super().__init__(net, loss_fn, face_detector, amplification)
        self.steps = hp["steps"]
        self.alpha = hp["alpha"]
        self.epsilon = hp["epsilon"]
        self.name = "pgd"

    def perturb(self, X, y, device=None):
        """ generates adversarial examples to given data points and labels (X, y) based on PGD approach. """
        original_X = X.clone()
        for i in range(self.steps):
            X.requires_grad_()
            outputs = self.net(X)
            _loss = self.loss_fn(outputs, y, the_original_input=X, the_pertubation=X - original_X,
                                 face_embedding=self.net).to(device)
            _loss.backward()

            X = X - self.alpha * X.grad.sign()
            if X.grad is not None:
                X.grad.zero_()
            diff = torch.clamp(X - original_X, min=-self.epsilon, max=self.epsilon)  # gradient projection
            X = torch.clamp(original_X + diff, min=-1.0,
                            max=1.0).detach_()  # to stay in image range [0,1] #TODO: check detach_ why _

            # print(_loss.item())

        return X
