import copy

import torch
from torch.nn import functional as F
# from RT4KSR.superresolution_for_saliency import SuperResolution
# superresolution = SuperResolution()
class CAMERAS():
    def __init__(self, model, targetLayerName, inputResolutions=None,use_superresolution=False):
        self.model = model
        self.inputResolutions = inputResolutions

        if self.inputResolutions is None:
            self.inputResolutions = list(range(224, 1000, 100))

        self.classDict = {}
        self.probsDict = {}
        self.featureDict = {}
        self.gradientsDict = {}
        self.targetLayerName = targetLayerName
        self.use_superresolution = use_superresolution
    def _recordActivationsAndGradients(self, inputResolution, image, classOfInterest=None):
        def forward_hook(module, input, output):
            self.featureDict[inputResolution] = (copy.deepcopy(output.clone().detach().cpu()))

        def backward_hook(module, grad_input, grad_output):
            self.gradientsDict[inputResolution] = (copy.deepcopy(grad_output[0].clone().detach().cpu()))

        for name, module in self.model.named_modules():
            if name == self.targetLayerName:
                forwardHandle = module.register_forward_hook(forward_hook)
                backwardHandle = module.register_backward_hook(backward_hook)
        device = image.device
        logits = self.model(image)
        softMaxScore = F.softmax(logits, dim=1)
        if logits.ndim == 4:
            logits = logits.mean(dim=(-1,-2))
        if softMaxScore.ndim == 4:
            softMaxScore = softMaxScore.mean(dim=(-1,-2))

        probs, classes = softMaxScore.sort(dim=1, descending=True)

        if classOfInterest is None:
            ids = classes[:, [0]]
        else:
            ids = torch.tensor(classOfInterest).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

        self.classDict[inputResolution] = ids.clone().detach().item()
        self.probsDict[inputResolution] = probs[0, 0].clone().detach().item()

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, ids, 1.0)

        self.model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=False)
        forwardHandle.remove()
        backwardHandle.remove()
        del forward_hook
        del backward_hook

    def _estimateSaliencyMap(self, classOfInterest,device='cpu'):
        saveResolution = self.inputResolutions[0]
        groundTruthClass = self.classDict[saveResolution]
        meanScaledFeatures = None
        meanScaledGradients = None

        count = 0
        for resolution in self.inputResolutions:
            if groundTruthClass == self.classDict[resolution] or self.classDict[resolution] == classOfInterest:
                count += 1
                upSampledFeatures = F.interpolate(self.featureDict[resolution].to(device), (saveResolution, saveResolution), mode='bilinear', align_corners=False)
                upSampledGradients = F.interpolate(self.gradientsDict[resolution].to(device), (saveResolution, saveResolution), mode='bilinear', align_corners=False)

                if meanScaledFeatures is None:
                    meanScaledFeatures = upSampledFeatures
                else:
                    meanScaledFeatures += upSampledFeatures

                if meanScaledGradients is None:
                    meanScaledGradients = upSampledGradients
                else:
                    meanScaledGradients += upSampledGradients

        meanScaledFeatures /= count
        meanScaledGradients /= count

        fmaps = meanScaledFeatures
        grads = meanScaledGradients

        saliencyMap = torch.mul(fmaps, grads).sum(dim=1, keepdim=True)

        saliencyMap = F.relu(saliencyMap)
        B, C, H, W = saliencyMap.shape
        saliencyMap = saliencyMap.view(B, -1)
        saliencyMap -= saliencyMap.min(dim=1, keepdim=True)[0]
        saliencyMap /= saliencyMap.max(dim=1, keepdim=True)[0]
        saliencyMap = saliencyMap.view(B, C, H, W)

        saliencyMap = torch.squeeze(torch.squeeze(saliencyMap, dim=0), dim=0)
        return saliencyMap

    def run(self, image, classOfInterest=None):
        device = image.device
        if self.use_superresolution:
            superresolution.to(device)
            superresolution.init_image(image,max(self.inputResolutions))
        self.shape0 = image.shape[-2:]
        for index, inputResolution in enumerate(self.inputResolutions):
            if index == 0:
                upSampledImage = image#.cuda()
            else:
                if not self.use_superresolution:
                    min_shape0 = min(self.shape0)
                    max_shape0 = max(self.shape0)
                    newShape = (self.shape0[0]/min_shape0)*inputResolution,(self.shape0[1]/min_shape0)*inputResolution
                    newShape = int(newShape[0]),int(newShape[1])
                    upSampledImage = F.interpolate(image, newShape, mode='bicubic', align_corners=False).to(device)
                else:
                    upSampledImage = superresolution.get_image_at_scale(inputResolution)
                    
                    print('check range of values and visualize.')


            self._recordActivationsAndGradients(inputResolution, upSampledImage, classOfInterest=classOfInterest)

        saliencyMap = self._estimateSaliencyMap(classOfInterest=classOfInterest,device=device)
        return saliencyMap

