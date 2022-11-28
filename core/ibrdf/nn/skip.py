import torch
import torch.nn as nn

class StackSequential(nn.ModuleList):
    def __init__(self):
        super(StackSequential, self).__init__()

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

class Concat(nn.Module):
    def __init__(self, dim, m1, m2):
        super(Concat, self).__init__()
        self.mDim = dim
        self.mModule1 = m1
        self.mModule2 = m2

    def forward(self, tensor):
        m1Output = self.mModule1(tensor)
        m2Output = self.mModule2(tensor)

        minShape2 = min(m1Output.size(2), m2Output.size(2))
        minShape3 = min(m1Output.size(3), m2Output.size(3))

        outputs = []
        if (
            (m1Output.size(2) == minShape2) and 
            (m2Output.size(2) == minShape2) and
            (m1Output.size(3) == minShape3) and 
            (m2Output.size(3) == minShape3)):
            outputs.append(m1Output)
            outputs.append(m2Output)
        else:
            diff2 = (m1Output.size(2) - minShape2) / 2
            diff3 = (m1Output.size(3) - minShape3) / 2
            outputs.append(m1Output[:,:,diff2:diff2+minShape2,diff3:diff3+minShape3])

            diff2 = (m2Output.size(2) - minShape2) / 2
            diff3 = (m2Output.size(3) - minShape3) / 2
            outputs.append(m2Output[:,:,diff2:diff2+minShape2,diff3:diff3+minShape3])

        return torch.cat(outputs, self.mDim)



class Skip(nn.Module):
    def __init__(
        self, 
        inFeatures, outFeatures, 
        numChannelsDown, numChannelsUp, numChannelsSkip,
        filterSizeDown = 3,
        filterSizeUp = 3,
        filterSizeSkip = 1,
        needBias = True,
        need1x1Up = True,
    ):
        super(Skip, self).__init__()
        padDown = ((filterSizeDown - 1) // 2)
        padUp = ((filterSizeUp - 1) // 2)
        padSkip = ((filterSizeSkip - 1) // 2)
        padSizeDown = padDown #[ padDown, padDown, padDown, padDown ]
        padSizeUp = padUp #[ padUp, padUp, padUp, padUp ]
        padSizeSkip = padSkip #[ padSkip, padSkip, padSkip, padSkip ]

        numScales = len(numChannelsDown)
        lastScale = numScales - 1
        self.model = StackSequential()
        modelTmp = self.model
        # StackSequentail modelTmp = model;

        actFn = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        inputDepth = inFeatures
        for i in range(numScales):
            deeper = StackSequential()
            skip = StackSequential()

            if (numChannelsSkip[i] != 0):
                modelTmp.append(Concat(1, skip, deeper))
            else:
                modelTmp.append(deeper)

            modelTmp.append(
                nn.BatchNorm2d(numChannelsSkip[i] + (numChannelsUp[i + 1] if (i < lastScale) else numChannelsDown[i]))
            )

            if numChannelsSkip[i] != 0:
                skip.append(
                    nn.Conv2d(
                        inputDepth, numChannelsSkip[i], filterSizeSkip, 
                        padding=padSizeSkip, padding_mode='reflect', 
                        bias=needBias
                    )
                )
                skip.append(nn.BatchNorm2d(numChannelsSkip[i]))
                skip.append(actFn)

            deeper.append(
                nn.Conv2d(
                    inputDepth, numChannelsDown[i], filterSizeDown, 
                    stride=2, padding=padSizeDown, padding_mode='reflect', 
                    bias=needBias
                )
            )
            deeper.append(
                nn.BatchNorm2d(numChannelsDown[i])
            )
            deeper.append(actFn)

            deeperMain = StackSequential()

            if i == lastScale:
                k = numChannelsDown[i]
            else:
                deeper.append(deeperMain)
                k = numChannelsUp[i + 1]

            deeper.append(
                nn.Upsample(
                    scale_factor=(2.0,2.0),
                    mode='bilinear',
                    align_corners=False,
                )
            )

            modelTmp.append(
                nn.Conv2d(
                    numChannelsSkip[i] + k, numChannelsUp[i], filterSizeUp, 
                    padding=padSizeUp, padding_mode='reflect', 
                    bias=needBias
                )
            )
            modelTmp.append(nn.BatchNorm2d(numChannelsUp[i]))
            modelTmp.append(actFn)

            if need1x1Up:
                modelTmp.append(
                    nn.Conv2d(
                        numChannelsUp[i], numChannelsUp[i], 1,
                        bias=needBias
                    )
                )
                modelTmp.append(nn.BatchNorm2d(numChannelsUp[i]))
                modelTmp.append(actFn)

            inputDepth = numChannelsDown[i]
            modelTmp = deeperMain

        self.model.append(nn.Conv2d(numChannelsUp[0], outFeatures, 1, bias=needBias))

    def forward(self, x):
        return torch.exp(self.model(x))

