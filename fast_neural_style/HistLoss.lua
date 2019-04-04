require "torch"
require "libcuda_utils"

-- Histogram loss from: https://arxiv.org/pdf/1701.08893.pdf
local HistLoss, parent = torch.class("nn.HistLoss", "nn.Module")

function HistLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.loss = 0
  self.nbins = 256
  self.minJ = {}
  self.maxJ = {}
  self.histJ = {}
  self.cumJ = {}
  self.err = nil
end

function HistLoss:updateOutput(input)
  local features = input
  local dtype = features:type()
  local N, C, H, W = features:size(1), features:size(2), features:size(3), features:size(4)
  if self.mode == "capture" then
    for i = 1, N do
      self.minJ[i] = features[{{i}, {}, {}, {}}]:view(C, H * W):min(2)
      self.maxJ[i] = features[{{i}, {}, {}, {}}]:view(C, H * W):max(2)
      self.histJ[i] =
        cuda_utils.histogram_nomask(
        features[{{i}, {}, {}, {}}]:cuda(),
        self.nbins,
        self.minJ[i]:cuda(),
        self.maxJ[i]:cuda()
      ):float()
      self.cumJ[i] = torch.cumsum(self.histJ[i], 2)
    end
  elseif self.mode == "loss" then
    self.loss = 0
    local R = features:clone()
    for i = 1, N do
      local sortI, idxI = torch.sort(features[{{i}, {}, {}, {}}]:view(C, H * W), 2)
      cuda_utils.hist_remap_nomask(
        features[{{i}, {}, {}, {}}],
        self.histJ[i]:cuda(),
        self.cumJ[i]:cuda(),
        self.minJ[i]:cuda(),
        self.maxJ[i]:cuda(),
        self.nbins,
        sortI:cuda(),
        idxI:cudaInt(),
        R[{{i}, {}, {}, {}}]
      )
    end
    self.err = features:clone():add(-1, R)
    self.loss = self.loss + self.err:pow(2.0):sum() * self.strength / features:nElement()
  end
  self.output = input
  return self.output
end

function HistLoss:updateGradInput(input, gradOutput)
  local features = input
  self.gradInput:resizeAs(input):zero()
  if self.mode == "capture" or self.mode == "none" then
    self.gradInput = gradOutput
  elseif self.mode == "loss" then
    self.gradInput:add(self.err)
    local magnitude = torch.norm(self.gradInput, 2)
    self.gradInput:div(magnitude + 1e-8)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  end
  return self.gradInput
end

function HistLoss:setMode(mode)
  if mode ~= "capture" and mode ~= "loss" and mode ~= "none" then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
