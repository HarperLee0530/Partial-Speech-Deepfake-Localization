import torch
from supervision import aggregate

def test_padding_empty_and_constant():
    z=torch.tensor([[2.,2.,2.,2.,2.,2.,2.,100.]+[4.]*8],requires_grad=True)
    d=torch.tensor([[.02]*6+[.01,0.]+[0.]*8])
    for mode in ("M0","R1"):
        got=aggregate(z,d,mode)
        assert torch.allclose(got,torch.tensor([[2.,0.]]))
        g,=torch.autograd.grad(got.sum(),z,retain_graph=True)
        assert torch.equal(g[d==0],torch.zeros_like(g[d==0]))
        assert torch.isfinite(g).all()

def test_full_partial_and_r1_derivative():
    z=torch.tensor([[1000.,-1000.,1.,2.,3.,4.,5.,6.]],dtype=torch.float64,requires_grad=True)
    d=torch.tensor([[.02,.02,.02,.01,0.,0.,0.,0.]],dtype=torch.float64)
    w=d/d.sum()
    m=aggregate(z,d,"M0")
    assert torch.allclose(m,(z*w).sum(-1))
    q=aggregate(z,d,"R1")
    g,=torch.autograd.grad(q.sum(),z)
    expected=torch.softmax((z+w.clamp_min(1e-30).log()).masked_fill(d==0,-torch.inf),-1)
    assert torch.allclose(g,expected)
    assert torch.isfinite(q).all()
    assert torch.equal(aggregate(z,d,"A0"),z.reshape(1,1,8).mean(-1))
    for mode in ("M0","R1"):
        x=torch.tensor([[3.]*8],requires_grad=True)
        assert torch.allclose(aggregate(x,torch.ones_like(x)*.02,mode),torch.tensor([[3.]]))

def test_total_objective_and_random_state():
    import supervision
    from cnn_ps.train_dual import training_loss as original
    from torch.nn import functional as F
    z=torch.arange(16,dtype=torch.float32).reshape(1,16).div(10).requires_grad_()
    batch=dict(fine_weights=torch.tensor([[.02]*10+[.01]+[0.]*5]),
               weights=torch.tensor([[.16,.05]]), fine_labels=torch.tensor([[1.]*6+[0.]*10]),
               labels=torch.tensor([[1.,0.]]), lengths=torch.tensor([16]),label_lengths=torch.tensor([2]))
    state=torch.random.get_rng_state().clone()
    supervision.ACTIVE_ARM="A0"
    a,_=supervision.training_loss(z,[],batch,"cpu",False)
    b,_=original(z,[],batch,"cpu",False)
    assert torch.equal(a,b)
    ga,=torch.autograd.grad(a,z,retain_graph=True)
    gb,=torch.autograd.grad(b,z,retain_graph=True)
    assert torch.equal(ga,gb)
    fine=(F.binary_cross_entropy_with_logits(z,batch["fine_labels"],reduction="none")*batch["fine_weights"]).sum()/batch["fine_weights"].sum()
    for mode in ("S1","M0","R1"):
        supervision.ACTIVE_ARM=mode
        got,_=supervision.training_loss(z,[],batch,"cpu",False)
        expected=fine if mode=="S1" else fine+supervision.coarse_loss(z,batch,mode)
        assert torch.allclose(got,expected,atol=1e-6)
        assert torch.isfinite(torch.autograd.grad(got,z,retain_graph=True)[0]).all()
    assert torch.equal(state,torch.random.get_rng_state())
    supervision.ACTIVE_ARM="A0"
