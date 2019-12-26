# 간단한 곱셈 계층 구현

class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  def forward(self,x,y):
    self.x = x
    self.y = y
    out = x * y
    return out
  def backward(self, dout): # dout은 미분
    dx = dout*self.y # x와 y를 바꾼다
    dy = dout*self.x
    return dx,dy