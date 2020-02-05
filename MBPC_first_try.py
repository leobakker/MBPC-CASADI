import casadi as ca
import matplotlib.pyplot as plt

class MPC ():
    """
    Leo Bakker 2020    
    
    MBPC for heating control of building
    control heating energy, 
    minimizing discomfort and energy use 
    
    Tout    Tin     Tmass
    *---R01--*--R12---*
             |        |
             =C1      =C2
             |        |
    
    dTin/C1 = (Tout - Tin)/R01 + (Tmass - Tin)/R12 + qint + qsol + qheating - qcooling 
    dTmass/C2 = (Tmass - Tin)/R12     
    
    x1' = ((Tout-x1)/R01 + (Tmass - Tin)/R12 + Qin + u)/C1
    x2' = ((x1-x2)/R01)/C2
   
    next steps:
    -1- include vectors for boundary conditions (e.g. Tout, Tset, Qint Qsolar) 
    -2- include multiple controlled actuators (e.g. heating + cooling + solar shading)
    -3- include temperature range (between min and max temperature) in penalty function 
    -4- test multiple optimizers 
    
    """
    
    def __init__(self):
        self.t = 24.            # Time horizon
        self.N = 100            # number of control intervals
        self.Tout = 15.         # outdoor temperature 
        self.R01 = 2.           # thermal resistance between outdoors and indoors
        self.R12 = 1            # thermal resistance between indoor and building mass 
        self.C1 = 5.            # thermal capacity indoor air 
        self.C2 = 10.           # thermal capacity building mass  
        self.Tset = 20.         # setpoint temperature 
        self.Qin = 1.           # heat entering e.g. internal heat / solar radiation entering 

    def model (self):
        x1 = ca.MX.sym('x1')    # indoor temperature
        x2 = ca.MX.sym('x2')    # outdoor temperature 
        x = ca.vertcat(x1, x2)  
        u = ca.MX.sym('u')      # heating energy 
        xdot = ca.vertcat(((self.Tout-x1)/self.R01 + (x2-x1)/self.R12 + self.Qin + u)/self.C1,(x1-x2)/(self.R12)/self.C2)  # Model equations
        L = (x1-self.Tset)**2 + 0.001*u**2              # Objective term  
        self.f = ca.Function('f', [x, u], [xdot, L])    # Formulate discrete time dynamics

    def rc (self):
        M = 4       # RK4 steps per interval # Fixed step Runge-Kutta 4 integrator
        DT = self.t/self.N/M
        X0 = ca.MX.sym('X0', 2)
        U = ca.MX.sym('U')
        X = X0
        Q = 0
        for j in range(M):
            k1, k1_q = self.f(X, U)
            k2, k2_q = self.f(X + DT/2 * k1, U)
            k3, k3_q = self.f(X + DT/2 * k2, U)
            k4, k4_q = self.f(X + DT * k3, U)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        self.F = ca.Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

    def NLP (self):
        # Start with an empty NLP
        w,w0,lbw,ubw=[],[],[],[]; J=0; g,lbg,ubg=[],[],[]
        # "Lift" initial conditions
        Xk = ca.MX.sym('X0', 2)
        w   += [Xk]
        lbw += [0, 1]
        ubw += [0, 1]
        w0  += [0, 1]
        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k))
            w   += [Uk]
            lbw += [-15]
            ubw += [15]
            w0  += [0]
            # Integrate till the end of the interval
            Fk = self.F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J=J+Fk['qf']
            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), 2)
            w   += [Xk]
            lbw += [-0.25, -ca.inf]
            ubw += [ca.inf, ca.inf]
            w0  += [0, 0]
            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0, 0]
            ubg += [0, 0]
        # Create an NLP solver
        prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
        solver = ca.nlpsol('solver', 'ipopt', prob);
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        self.w_opt = sol['x'].full().flatten()

    def plott(self):    # Plot the solution
        x1_opt = self.w_opt[0::3]
        x2_opt = self.w_opt[1::3]
        u_opt = self.w_opt[2::3]
        tgrid = [self.t/self.N*k for k in range(self.N+1)]
        plt.figure(1)
        plt.clf()
        plt.plot(tgrid, x1_opt, '--')
        plt.plot(tgrid, x2_opt, '-')
        plt.step(tgrid, ca.vertcat(ca.DM.nan(1), u_opt), '-.')
        plt.xlabel('t')
        plt.legend(['Temperature room','Temperature building mass','Heating power'])
        plt.grid()
        plt.show()

mpc = MPC()
mpc.model()
mpc.rc()
mpc.NLP()
mpc.plott()
