import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(42)
np.random.seed(42)

# 1.VARIABLES, CLASSES AND FUNCTIONS ------------------------------------------------------------

# 1.1.Variables ---------------------------------------------------------------------------------

d = 2                                       #number of dimensions use in this example

c_np = np.array([[1.0, 0.3],                #instantaneous cov matrix c (numpy)
                 [0.3, 0.8]])
c = torch.tensor(c_np, dtype=torch.float64) #we need c in PyTorch tensor for later computations

Sigma_np = np.array([[2.0, 0.5],            #stationary cov \Sigma (numpy)
                     [0.5, 1.5]])
Sigma     = torch.tensor(Sigma_np,                dtype=torch.float64) #same as with c
Sigma_inv = torch.tensor(np.linalg.inv(Sigma_np), dtype=torch.float64) #same for \Sigma^{-1}


Q = torch.tensor(np.linalg.cholesky(Sigma_np), dtype=torch.float64) #Cholesky decomposition of 
                                            #\Sigma=QQ^T for later sampling from N(0,\Sigma)
g_star = 0.125 * np.trace(np.linalg.inv(Sigma_np) @ c_np) #analytical growth rate from Part I ex.7


# 1.2.Agent and Adversary Networks --------------------------------------------------------------

class AgentNet(nn.Module):

    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        ).double()

    def forward(self, x):
        return self.net(x).squeeze(-1)

class Adversary(nn.Module):

    def __init__(self):
        super().__init__()
        self.s12 = nn.Parameter(torch.randn(1, dtype=torch.float64))

    def get_B(self):
        S = torch.zeros((d, d), dtype=torch.float64)
        S[0, 1] =  self.s12
        S[1, 0] = -self.s12
        return -0.5 * c @ Sigma_inv + S @ Sigma_inv


# 1.3.Functions ---------------------------------------------------------------------------------

def p_sample(n):
    z = torch.randn(n, d, dtype=torch.float64)
    return z @ Q.T

def grad(phi_net, x):
    x = x.clone().detach().requires_grad_(True)
    phi = phi_net(x)
    gradient = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
    return gradient

def g_est(agent, adv, x): #growth rate estimator, eq.(20)
    gradient = grad(agent, x)
    B    = adv.get_B()
    drift_term = (gradient * (x @ B.T)).sum(dim=1)
    quad_term  = (gradient @ c * gradient).sum(dim=1)
    return -drift_term.mean() - 0.5 * quad_term.mean()


# 2.TRAINING -----------------------------------------------------------------------------------

print(f"Analytical growth rate: g* = {g_star:.6f}")

agent = AgentNet(d)
adv = Adversary()

agent_optim = optim.Adam(agent.parameters(), lr=1e-3)
adv_optim = optim.Adam(adv.parameters(), lr=1e-3)

n_epochs = 3000
N = 2048
history = []

for epoch in range(n_epochs):
    x = p_sample(N)

    # Adversary training, min g_est
    adv_optim.zero_grad()
    g = g_est(agent, adv, x)
    g.backward()
    nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
    adv_optim.step()

    # Agent training, max g_est
    agent_optim.zero_grad()
    g = g_est(agent, adv, x)
    (-g).backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    agent_optim.step()

    history.append(float(g.detach()))
    if epoch % 500 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:4d} | g = {g.item():.6f} | g* = {g_star:.6f}")


# 3.RESULTS AND PLOTS --------------------------------------------------------------------------

g_learned = history[-1]
print("\nFinal Results")
print(f"Learned growth rate : {g_learned:.6f}")
print(f"Analytical g*       : {g_star:.6f}")

# 3.1.Evaluation grid --------------------------------------------------------------------------

grid_pts = 20
xs = np.linspace(-3, 3, grid_pts)
ys = np.linspace(-3, 3, grid_pts)
Xg, Yg   = np.meshgrid(xs, ys)
grid_np  = np.stack([Xg.ravel(), Yg.ravel()], axis=1)   # (M, 2)
grid     = torch.tensor(grid_np, dtype=torch.float64)

grid_req     = grid.clone().requires_grad_(True)
phi_vals     = agent(grid_req)
learned_grad = torch.autograd.grad(phi_vals.sum(), grid_req)[0].detach().numpy() #learned strategy

sign         = np.sign(np.mean(np.sum(learned_grad * (-grid_np), axis=1)))
learned_grad = sign * learned_grad

Sigma_inv_np     = np.linalg.inv(Sigma_np)
analytical_grad  = -0.5 * (grid_np @ Sigma_inv_np.T) #analytical strategy from Part I ex.6 eq.(11)

U_l = learned_grad[:, 0].reshape(grid_pts, grid_pts) #reshaping to the grid
V_l = learned_grad[:, 1].reshape(grid_pts, grid_pts)
U_a = analytical_grad[:, 0].reshape(grid_pts, grid_pts)
V_a = analytical_grad[:, 1].reshape(grid_pts, grid_pts)


# 3.2.Plots: training curve | learned strategy | analytical strategy ----------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3.2.1.Training curve --------------------------------------------------------
window   = 50
smoothed = np.convolve(history, np.ones(window) / window, mode='valid')
axes[0].plot(history, lw=0.5, alpha=0.3, color="steelblue")
axes[0].plot(range(window - 1, len(history)), smoothed, lw=1.5,
             color="steelblue", label="Learned $g$ (smoothed)")
axes[0].axhline(g_star, color="red", lw=1.5, linestyle="--",
                label=r"Analytical $g^*$")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Growth rate")
axes[0].set_title("Minimax training")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 3.2.2.Learned strategy ------------------------------------------------------
axes[1].quiver(Xg, Yg, U_l, V_l, color="steelblue", alpha=0.8)
axes[1].set_xlabel("$x_1$")
axes[1].set_ylabel("$x_2$")
axes[1].set_title(r"Learned strategy $\nabla\varphi_\alpha$")
axes[1].grid(True, alpha=0.3)

# 3.2.3.Analytical strategy ---------------------------------------------------
axes[2].quiver(Xg, Yg, U_a, V_a, color="red", alpha=0.8)
axes[2].set_xlabel("$x_1$")
axes[2].set_ylabel("$x_2$")
axes[2].set_title(r"Analytical strategy $-\frac{1}{2}\Sigma^{-1}x$")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("robust_growth_rate.png", dpi=150)
plt.show()