#!/usr/bin/env python3
"""
Hit.py v5.1 — Tensor-Flower Comet Redirection System (Interactive Monolith)
============================================================================
Browser dashboard with tabs: SCOPE · IMPACT · 3D VIEW · STATS · CONSOLE
Hotkeys 1-5, ESC info overlay, hoverable gate spheres, celestial body orbits.
Physics: 2-body gravity, RK4, Newton shooting, 12-gate Jacobian Dv corrections.
"""
import numpy as np, http.server, threading, webbrowser, json, argparse, sys
from datetime import datetime
np.random.seed(42)

MU=4.0*np.pi**2; HIT_RAD=0.02; DT=0.005; DAMP=0.7; MAX_DV=0.05
B_DEF=(-0.6,-1.0392);TG_DEF=(1.52,0.0)
# Default projectile spec — every barrel carries one so 'energy needed to hit
# the target' is part of the default barrel knowledge. Overridable per barrel.
# 100 m rocky body at 2.6 t/m^3 ≈ 1.36e9 kg; chemical Isp 350 s (solid kicker).
DEFAULT_PROJECTILE=dict(diameter_m=100.0,density_kg_m3=2600.0,Isp_s=350.0)

# ═══════════════════════════════════════════════════════════════════════
# FTOP vs. central mass — a conceptual split:
#   FTOP = (0,0) is the *orientation anchor* + center of the Flower-of-Life
#   template. The Sun (and any other gravitating body) lives at its own
#   world-AU position. By default, Sun sits at FTOP for backward-compatibility
#   with the 2-body Newton-shooter, but it can be placed anywhere.
#
# BODIES is a module-global list of (x, y, mu) tuples consumed by _g().
# Any solver / propagator that imports _g picks up the full n-body field.
# When the list contains just one entry (Sun at FTOP, mu = MU), the model
# collapses to the original 2-body Keplerian gravity.
# ═══════════════════════════════════════════════════════════════════════
BODIES=[(0.0,0.0,MU)]  # default: Sun at FTOP, μ = 4π² AU³/yr²
# Vectorised cache — the RK4 integrator calls _g() millions of times, so we
# keep a numpy view of BODIES (positions + μs) that's rebuilt whenever BODIES
# is reassigned. Access via _BODIES_XY / _BODIES_MU; refresh via _refresh_bodies().
_BODIES_XY=np.array([[0.0,0.0]]);_BODIES_MU=np.array([MU])
# Fast-path cache for the single-body case (common default: Sun at FTOP).
# When exactly one body exists, skip the Python for-loop entirely and use a
# direct scalar formula. This is the hot path for 99% of default runs and
# halves _swarm() / _solve_from() cost.
_BODIES_SINGLE=True;_SB_X=0.0;_SB_Y=0.0;_SB_MU=MU
def _refresh_bodies():
    global _BODIES_XY,_BODIES_MU,_BODIES_LOCAL,_BODIES_SINGLE,_SB_X,_SB_Y,_SB_MU
    _BODIES_XY=np.array([[b[0],b[1]] for b in BODIES],dtype=float)
    _BODIES_MU=np.array([b[2] for b in BODIES],dtype=float)
    _BODIES_LOCAL=list(BODIES)  # local copy so _g's loop doesn't hit module globals
    _BODIES_SINGLE=(len(BODIES)==1)
    if _BODIES_SINGLE:
        _SB_X,_SB_Y,_SB_MU=BODIES[0]

# ═══════════════════════════════════════════════════════════════════════
# Solar system — 1 Sun + 20 planets + random moons per planet.
# Reproducible via `np.random.seed(42)` above. The Sun's μ is the standard
# heliocentric 4π². Planet μ is sampled log-uniformly across 1e-10 … 1e-3 × μ,
# covering asteroid-class through Jupiter-class bodies. Moon μ is 1e-4 … 1e-2
# of its parent planet.
#
# All orbits are taken as coplanar circles in the XY scope plane (the same
# plane the corridor lives in), so the perturbation on the projectile is
# in-plane and easy to reason about.
#
# `solar_system_snapshot(t)` freezes every planet + moon at its phase-t
# position, returning a list of (x, y, mu) tuples suitable for BODIES.
# ═══════════════════════════════════════════════════════════════════════
def _build_solar_system(n_planets=20,seed=42):
    rng=np.random.RandomState(seed)
    sun={'name':'Sun','mu':MU,'r_orbit':0.0,'period':1e18,
         'phase':0.0,'color':'#ffcc00','r_vis':0.06,
         'parent':-1,'moons':[]}
    bodies=[sun]
    # Planet orbital radii span 0.39 AU (Mercury-like) → 40 AU (Pluto-like)
    # spaced in log-r so the inner solar system gets finer detail.
    r_planets=np.logspace(np.log10(0.39),np.log10(40.0),n_planets)
    for i,r in enumerate(r_planets):
        # Planet μ  log-uniform  in [1e-9 μ_sun, 1e-3 μ_sun]
        pmu=MU*10**rng.uniform(-9,-3)
        phase=rng.uniform(0,2*np.pi)
        period=2*np.pi*(r**1.5)/np.sqrt(MU)  # Kepler T = 2π √(a³/μ)
        pcol=['#88aaff','#ff8866','#aaff88','#ffaa44','#cc44ff',
              '#44eecc','#eeaa88','#8844cc','#ccff44','#ff4488'][i%10]
        nmoons=int(rng.randint(0,6))  # 0 .. 5 moons
        moons=[]
        for j in range(nmoons):
            # Moon orbits planet at 0.002 .. 0.03 AU; sample log-uniform
            mr=10**rng.uniform(-2.7,-1.5)
            mmu=pmu*10**rng.uniform(-4,-2)
            mphase=rng.uniform(0,2*np.pi)
            mperiod=2*np.pi*(mr**1.5)/np.sqrt(max(pmu,1e-12))
            moons.append({'name':f'P{i+1}-m{j+1}','mu':mmu,'r_orbit':mr,
                          'period':mperiod,'phase':mphase,'parent':i+1,
                          'color':'#aaaaaa','r_vis':0.012})
        bodies.append({'name':f'Planet{i+1:02d}','mu':pmu,'r_orbit':float(r),
                       'period':float(period),'phase':phase,'parent':0,
                       'color':pcol,'r_vis':0.035,'moons':moons})
    return bodies

SOLAR=_build_solar_system()

def solar_system_snapshot(t,solar=None,include_moons=True):
    """(x, y, mu) list for every body at time t (years, from epoch 0)."""
    if solar is None: solar=SOLAR
    out=[]
    for i,b in enumerate(solar):
        if i==0:
            out.append((0.0,0.0,b['mu']))  # Sun at FTOP by default
        else:
            th=b['phase']+2*np.pi*t/max(b['period'],1e-12)
            x=b['r_orbit']*np.cos(th);y=b['r_orbit']*np.sin(th)
            out.append((x,y,b['mu']))
            if include_moons:
                for m in b.get('moons',[]):
                    mth=m['phase']+2*np.pi*t/max(m['period'],1e-12)
                    mx=x+m['r_orbit']*np.cos(mth);my=y+m['r_orbit']*np.sin(mth)
                    out.append((mx,my,m['mu']))
    return out

def _serialise_solar(solar=None):
    """Lightweight JSON representation of the solar system for the browser."""
    if solar is None: solar=SOLAR
    out=[]
    for i,b in enumerate(solar):
        e=dict(idx=i,name=b['name'],mu=float(b['mu']),r_orbit=float(b['r_orbit']),
               period=float(b['period']),phase=float(b['phase']),
               color=b['color'],r_vis=float(b['r_vis']),parent=int(b.get('parent',-1)))
        mo=[]
        for j,m in enumerate(b.get('moons',[])):
            mo.append(dict(idx=j,name=m['name'],mu=float(m['mu']),
                           r_orbit=float(m['r_orbit']),period=float(m['period']),
                           phase=float(m['phase']),color=m['color'],r_vis=float(m['r_vis']),
                           parent=i))
        e['moons']=mo
        out.append(e)
    return out

_BODIES_LOCAL=[(0.0,0.0,MU)]  # a flat Python list for _g's hot loop (avoids global lookup)
def _g(r):
    """Total gravitational acceleration at position r, summed over BODIES.

    Hot path for the RK4 integrator (called ~10⁶× per MC campaign). Three
    code paths, in order of frequency:
      1. _BODIES_SINGLE  → scalar arithmetic, no loop, no np.array alloc (~3× faster)
      2. n-body          → Python for-loop (fast enough at N≈27 bodies)
      3. empty           → zero vector
    """
    if _BODIES_SINGLE:
        dx=_SB_X-r[0];dy=_SB_Y-r[1]
        n2=dx*dx+dy*dy
        if n2<1e-24:return np.zeros(2)
        inv=_SB_MU/(n2*n2**0.5)
        return np.array([inv*dx,inv*dy])
    ax=0.0;ay=0.0;rx=r[0];ry=r[1]
    for bx,by,bmu in _BODIES_LOCAL:
        dx=bx-rx;dy=by-ry
        n2=dx*dx+dy*dy
        if n2<1e-24: continue
        inv=bmu/(n2*n2**0.5)
        ax+=inv*dx;ay+=inv*dy
    return np.array([ax,ay])

# Scalar-tuple variant used by the fused RK4 fast path. Returns (ax, ay) with
# no np.array allocation so the 2-body hot loop avoids ~1.8M numpy alloc/frees
# per swarm solve.
def _g_xy(rx,ry):
    if _BODIES_SINGLE:
        dx=_SB_X-rx;dy=_SB_Y-ry
        n2=dx*dx+dy*dy
        if n2<1e-24:return(0.0,0.0)
        inv=_SB_MU/(n2*n2**0.5)
        return(inv*dx,inv*dy)
    ax=0.0;ay=0.0
    for bx,by,bmu in _BODIES_LOCAL:
        dx=bx-rx;dy=by-ry
        n2=dx*dx+dy*dy
        if n2<1e-24: continue
        inv=bmu/(n2*n2**0.5)
        ax+=inv*dx;ay+=inv*dy
    return(ax,ay)
def _rk4(s,h):
    """Pure-numpy RK4 step (kept for callers that want an array interface)."""
    def f(s): return np.array([s[2],s[3],*_g(s[:2])])
    k1=f(s);k2=f(s+.5*h*k1);k3=f(s+.5*h*k2);k4=f(s+h*k3)
    return s+(h/6)*(k1+2*k2+2*k3+k4)

def _rk4_scalar(x,y,vx,vy,h):
    """Scalar RK4 — no np.array allocations in the inner loop. About 3-4×
    faster than _rk4 for the common 2-body case because the hot _g_xy()
    returns a tuple, and state is carried as four Python floats.
    """
    # k1
    a1x,a1y=_g_xy(x,y)
    # k2 at (x+0.5h*vx, y+0.5h*vy, vx+0.5h*a1x, vy+0.5h*a1y)
    hx=0.5*h
    x2=x+hx*vx;y2=y+hx*vy
    a2x,a2y=_g_xy(x2,y2)
    # k3 at (x+0.5h*(vx+0.5h*a1x), ...)
    vx2=vx+hx*a1x;vy2=vy+hx*a1y
    x3=x+hx*vx2;y3=y+hx*vy2
    a3x,a3y=_g_xy(x3,y3)
    # k4 at (x+h*(vx+0.5h*a2x), ...)
    vx3=vx+hx*a2x;vy3=vy+hx*a2y
    x4=x+h*vx3;y4=y+h*vy3
    a4x,a4y=_g_xy(x4,y4)
    # accumulate — weighted sum of the four RK4 slopes
    vx4=vx+h*a3x;vy4=vy+h*a3y
    h6=h/6.0
    nx=x+h6*(vx+2*vx2+2*vx3+vx4)
    ny=y+h6*(vy+2*vy2+2*vy3+vy4)
    nvx=vx+h6*(a1x+2*a2x+2*a3x+a4x)
    nvy=vy+h6*(a1y+2*a2y+2*a3y+a4y)
    return nx,ny,nvx,nvy

def prop(st,T):
    """Full trajectory — returns an (N+1, 4) array of states."""
    n=max(1,round(T/DT));h=T/n
    x,y,vx,vy=float(st[0]),float(st[1]),float(st[2]),float(st[3])
    out=np.empty((n+1,4));out[0,0]=x;out[0,1]=y;out[0,2]=vx;out[0,3]=vy
    for i in range(n):
        x,y,vx,vy=_rk4_scalar(x,y,vx,vy,h)
        out[i+1,0]=x;out[i+1,1]=y;out[i+1,2]=vx;out[i+1,3]=vy
    return out

def pf(st,T):
    """Propagate-forward — returns only the final state (length-4 array)."""
    n=max(1,round(T/DT));h=T/n
    x,y,vx,vy=float(st[0]),float(st[1]),float(st[2]),float(st[3])
    for _ in range(n):
        x,y,vx,vy=_rk4_scalar(x,y,vx,vy,h)
    return np.array([x,y,vx,vy])
def stm(st,dt):
    r=st[:2];rn=np.linalg.norm(r)
    if rn<1e-12: return np.eye(4)
    r3,r5=rn**3,rn**5
    Uxx=-MU/r3+3*MU*r[0]**2/r5;Uyy=-MU/r3+3*MU*r[1]**2/r5;Uxy=3*MU*r[0]*r[1]/r5
    P=np.eye(4);P[0,2]=dt;P[1,3]=dt;P[2,0]=Uxx*dt;P[2,1]=Uxy*dt;P[3,0]=Uxy*dt;P[3,1]=Uyy*dt
    return P

class System:
    def __init__(self,ns=2000):
        self.ns=ns;self.Rs=1.2;self.Rs_base=1.2
        self.ba=240*np.pi/180  # TRUE 7 o'clock (240 deg math)
        self.bp=self.Rs*np.array([np.cos(self.ba),np.sin(self.ba)])
        self.tp=np.array([1.52,0.0]);self.tf=0.70;self.dtg=self.tf/13.0
        # tz is the target's Z-axis coordinate. The physics engine operates on
        # the 2D scope plane (XY), but the 3D display renders the target at
        # (tx, ty, tz) so off-plane targets are legal and visible. Physics
        # treats the XY projection as the active target; tz>0 means the
        # projectile still has to "close the gap" in-plane and the 3D views
        # show a vertical line reminding the user a Z-component exists.
        self.tz=0.0
        # Sun z (default same plane as everything else).
        self.sun_z=0.0
        self.scov=np.diag([1e-4,1e-4,1e-6,1e-6])
        # FTOP is the orientation anchor (0,0). The Sun's WORLD-AU position
        # can be set independently via self.sun_pos; by default Sun = FTOP
        # so the 2-body behaviour is preserved. The solar system is installed
        # into BODIES before every solve so planet perturbations propagate.
        self.sun_pos=np.array([0.0,0.0])
        # N-body perturbations OFF by default so initial launch is snappy. When
        # enabled (via CONFIG), the RK4 integrator sums gravity over all 27
        # solar-system bodies, which is ~25× slower per substep. Users who need
        # the full perturbed solution can toggle it on after inspecting the
        # baseline 2-body answer.
        self.use_perturbations=False
        self.solar=SOLAR
        # Cache key so _install_bodies() only rebuilds BODIES when state changes.
        self._bodies_cache_key=None
        # Auto-grow: the scope template is a scale unit. If any placed target/barrel
        # falls outside it, the stencil stretches uniformly so the whole scene fits.
        # Same geometry, different diameter — enables mapping galactic / mass-cluster scales.
        self._autogrow_Rs()
        # 12 clock-face gate positions (1 o'clock=60deg, 2=30deg, ... 12=90deg)
        clk=[((90-i*30)%360)*np.pi/180 for i in range(1,13)]
        self.gxy=self.Rs*np.column_stack(([np.cos(a) for a in clk],[np.sin(a) for a in clk]))
        self.log=[];self._p("="*62)
        self._p("  Hit.py v5.1 \u2014 Tensor-Flower Comet Redirection System")
        self._p("="*62)
        self._p("  Solving transfer orbit (Newton shooting)...")
        self.v0=self._solve()
        self._p("  Precomputing 12-gate correction Jacobians...")
        self.gs,self.gJ=self._gates()
        self.T=self._tensor()
        nm=np.linalg.norm(pf(np.array([*self.bp,*self.v0]),self.tf)[:2]-self.tp)
        v0m=np.linalg.norm(self.v0)
        self._p("-"*62)
        self._p("  SYSTEM CONFIGURATION")
        self._p("-"*62)
        self._p(f"  Barrel   : ({self.bp[0]:+.4f},{self.bp[1]:+.4f}) AU [7 o'clock, 240\u00b0]")
        self._p(f"  Target   : ({self.tp[0]:+.4f},{self.tp[1]:+.4f}) AU [outer ref orbit]")
        self._p(f"  v0       : ({self.v0[0]:+.4f},{self.v0[1]:+.4f}) AU/yr  |v0|={v0m:.4f}")
        self._p(f"  Miss     : {nm:.2e} AU")
        self._p(f"  Flight   : {self.tf} yr | Gates:12 | dt_gate: {self.dtg:.5f} yr")
        self._p(f"  Sims     : {ns:,} per campaign")
        self._p(f"  GM(\u03bc)   : {MU:.6f} AU\u00b3/yr\u00b2")
        self._p(f"  Hit rad  : {HIT_RAD} AU (~{HIT_RAD*1.496e8:.0f} km)")
        grow_x=self.Rs/self.Rs_base
        self._p(f"  Scope Rs : {self.Rs:.4f} AU  | Diameter: {2*self.Rs:.4f} AU  | Inner: {self.Rs*.58:.4f} AU  | Scale: {grow_x:.2f}x base ({self.Rs_base} AU)")
        self._p("-"*62)
        self._p("  GATE JACOBIAN SUMMARY")
        self._p("-"*62)
        self._p(f"  {'Gate':>4}  {'Position':>22}  {'Speed':>8}  {'R(FTOP)':>8}  {'J-cond':>8}")
        for k in range(12):
            gs=self.gs[k];spd=np.linalg.norm(gs[2:]);rsun=np.linalg.norm(gs[:2])
            Jv=self.gJ[k][:,2:]
            try: jc=np.linalg.cond(Jv)
            except: jc=9999
            self._p(f"  {k+1:4d}  ({gs[0]:+.4f},{gs[1]:+.4f})  {spd:8.4f}  {rsun:8.4f}  {min(jc,9999):8.1f}")
        self._p("-"*62)
        self._p("  RELATIVE TENSOR MATRIX [T] (normalized STM)")
        self._p("-"*62)
        labels=['  x ','  y ',' vx ',' vy ']
        self._p("       "+' '.join(f'{l:>8}' for l in labels))
        for i in range(4):
            row=' '.join(f'{self.T[i,j]:+8.4f}' for j in range(4))
            self._p(f"  {labels[i]} {row}")
        det=np.linalg.det(self.T);tr=np.trace(self.T)
        self._p(f"  det(T)={det:.6f}  tr(T)={tr:.6f}  ||T||_F={np.linalg.norm(self.T):.6f}")
        self._p("-"*62)
        self._p("  FLOWER-OF-LIFE CROSS-SECTION PROOF")
        self._p("-"*62)
        n_inner=7;n_scope=12;n_outer=12;n_total=n_inner+n_scope+n_outer
        max_pairs=n_total*(n_total-1)//2
        self._p(f"  Circles: {n_inner} inner + {n_scope} scope + {n_outer} outer = {n_total}")
        self._p(f"  Max pairwise tests: C({n_total},2) = {max_pairs}")
        self._p(f"  Each intersection yields 0 or 2 points (vesica piscis)")
        self._p(f"  Points deduplicated within \u03b5=0.02 AU")
        self._p(f"  Sorted: radial distance from FTOP, then CW from 12 o'clock")
        self._p(f"  Fibonacci indices {{0,1,2,3,5,8,13,21,34,...}} \u2192 bold rings")
        self._p(f"  Pattern: F(n) = F(n-1) + F(n-2), cross-sections spiral outward")
        self._p(f"  Proof: N intersections covers all vesica piscis loci of the")
        self._p(f"         Flower-of-Life lattice. No empty variables exist when")
        self._p(f"         all {n_total} circles are tested pairwise ({max_pairs} pairs).")
        self._p("="*62)
    def _p(self,m): self.log.append(m);print(m)
    def _autogrow_Rs(self,snap_barrel=True,include_swarm=True):
        """Resize the scope so the whole scene fits inside one template.

        The flower-of-life stencil is a *scale unit*, not a tile. When a target
        (or any barrel) is placed outside the current scope, the entire stencil
        grows uniformly — same 31-circle geometry, same 12 gate positions, just
        a larger diameter covering more AU. No extra spheres are added.

        `include_swarm=True` takes swarm free-placement positions into account
        so a target placed close but with far-out swarm barrels still fits.
        `snap_barrel=True` re-snaps the primary barrel to the new ring radius;
        set False to keep a custom (free-placed) barrel where it is.
        """
        r_ref=1.52  # outer reference orbit
        r_t=float(np.linalg.norm(self.tp))
        r_b=float(np.linalg.norm(self.bp))
        r_need=max(r_t,r_b,r_ref)
        if include_swarm:
            for p in (getattr(self,'swm_positions',None) or []):
                if p is None: continue
                r_need=max(r_need,float(np.linalg.norm(np.asarray(p,dtype=float))))
        # Scene must sit inside the template with a small margin
        margin=1.10
        step=self.Rs_base
        new_Rs=step*max(1,int(np.ceil((r_need*margin)/step)))
        # Enlarge OR shrink — always snap Rs to the scale needed. Shrinking
        # (when the user moves target back inside the base unit) returns the
        # scope to its smallest quantised size so the view is never needlessly
        # zoomed out.
        self.Rs=max(self.Rs_base,new_Rs)
        # Rebuild the 12 clock-face gate positions on the (possibly new) ring
        # so the scope-ring geometry stays locked to Rs.
        clk=[((90-i*30)%360)*np.pi/180 for i in range(1,13)]
        self.gxy=self.Rs*np.column_stack(([np.cos(a) for a in clk],[np.sin(a) for a in clk]))
        if snap_barrel:
            self.bp=self.Rs*np.array([np.cos(self.ba),np.sin(self.ba)])
    def _install_bodies(self,t0=0.0):
        """Push a snapshot of the solar system into the module-global BODIES
        list so the RK4 integrator picks up every gravitating source.

        Snapshot is frozen at t=t0 (typically 0) — planets don't move during
        the 0.7-year flight; this is the standard "restricted" n-body trick.
        The error vs. a full n-body propagation of the same scene is
        ≲ (period_planet / T_f)⁻¹, i.e. well below the 10⁻¹⁰ AU RK4 noise
        floor for outer planets; only inner-planet close passes would matter
        and the Newton shooter absorbs that into v₀.

        When `self.use_perturbations=False`, only the Sun is installed so the
        behaviour collapses back to the classic 2-body solution.
        """
        global BODIES
        sx,sy=float(self.sun_pos[0]),float(self.sun_pos[1])
        # Skip rebuild when the state hasn't changed — _solve/_gates/_parallel
        # all call _install_bodies(), which used to re-materialise the same
        # list N times per campaign.
        key=(sx,sy,bool(self.use_perturbations),float(t0),id(self.solar))
        if key==self._bodies_cache_key: return
        self._bodies_cache_key=key
        if not self.use_perturbations:
            BODIES=[(sx,sy,MU)]
        else:
            snap=solar_system_snapshot(t0,self.solar,include_moons=True)
            # snap[0] is the sun at (0,0,MU) by convention; translate to sun_pos.
            BODIES=[(sx+bx,sy+by,bmu) for (bx,by,bmu) in snap]
        _refresh_bodies()
    def _solve(self):
        # Ensure the n-body field is live before shooting.
        self._install_bodies()
        rb=np.linalg.norm(self.bp);vc=np.sqrt(MU/rb)
        rh=self.bp/rb;th=np.array([-rh[1],rh[0]]);bv=None;bm=1e30
        for sf in np.linspace(0.7,1.5,17):
            for ao in np.linspace(-0.8,0.8,17):
                v=vc*sf*(th*np.cos(ao)+rh*np.sin(ao))
                m=np.linalg.norm(pf(np.array([*self.bp,*v]),self.tf)[:2]-self.tp)
                if m<bm:bm=m;bv=v.copy()
        self._p(f"    Grid best: {bm:.4f} AU");v0=bv
        for it in range(120):
            f=pf(np.array([*self.bp,*v0]),self.tf);miss=f[:2]-self.tp
            if np.linalg.norm(miss)<1e-11: self._p(f"    Newton converged iter {it+1}");return v0
            J=np.zeros((2,2));eps=1e-8
            for j in range(2):
                vp=v0.copy();vp[j]+=eps;J[:,j]=(pf(np.array([*self.bp,*vp]),self.tf)[:2]-f[:2])/eps
            try:v0+=np.linalg.solve(J,-miss)
            except:v0+=np.random.normal(0,0.01,2)
        self._p(f"    Warning: miss={np.linalg.norm(miss):.2e}");return v0
    def _gates(self):
        self._install_bodies()
        s=np.array([*self.bp,*self.v0]);st=[];jc=[]
        for k in range(12):
            s=pf(s,self.dtg);st.append(s.copy())
            rem=self.tf-(k+1)*self.dtg
            if rem<1e-4:jc.append(np.zeros((2,4)));continue
            f0=pf(s,rem)[:2];J=np.zeros((2,4));eps=1e-8
            for j in range(4):
                sp=s.copy();sp[j]+=eps;J[:,j]=(pf(sp,rem)[:2]-f0)/eps
            jc.append(J)
        return st,jc
    def _tensor(self):
        s=np.array([*self.bp,*self.v0]);P=np.eye(4)
        for _ in range(13):P=stm(s,self.dtg)@P;s=pf(s,self.dtg)
        T=0.5*(P+P.T);d=np.sqrt(np.abs(np.diag(T)))+1e-15;return T/np.outer(d,d)
    def _fly(self,ptb,cor=False):
        s=np.array([*self.bp,*self.v0])+ptb
        if not cor:return pf(s,self.tf)[:2]
        for k in range(12):
            s=pf(s,self.dtg);n=np.random.multivariate_normal(np.zeros(4),self.scov)
            d=(s+n)-self.gs[k];pm=self.gJ[k]@d;Jv=self.gJ[k][:,2:]
            det=np.linalg.det(Jv)
            if abs(det)>1e-12:
                gd=DAMP*max(0.2,(12-k)/12.0);dv=np.linalg.solve(Jv,-pm)*gd
                dn=np.linalg.norm(dv)
                if dn>MAX_DV:dv*=MAX_DV/dn
                s[2:]+=dv
        return pf(s,self.dtg)[:2]
    def run(self):
        sig=np.array([0.012,0.012,0.003,0.003])
        bi=np.zeros((self.ns,2));ci=np.zeros((self.ns,2));tk=max(1,self.ns//5)
        self._p("  [1/2] Baseline campaign...")
        for i in range(self.ns):
            bi[i]=self._fly(np.random.normal(0,sig),False)
            if(i+1)%tk==0:sys.stdout.write(f"\r        {i+1}/{self.ns}");sys.stdout.flush()
        print()
        self._p("  [2/2] 12-gate corrected campaign...")
        for i in range(self.ns):
            ci[i]=self._fly(np.random.normal(0,sig),True)
            if(i+1)%tk==0:sys.stdout.write(f"\r        {i+1}/{self.ns}");sys.stdout.flush()
        print()
        bm=np.linalg.norm(bi-self.tp,axis=1);cm=np.linalg.norm(ci-self.tp,axis=1)
        r=dict(ts=datetime.now().isoformat(),n=self.ns,
            bh=round(float(np.mean(bm<HIT_RAD)*100),2),ch=round(float(np.mean(cm<HIT_RAD)*100),2),
            bmm=round(float(np.mean(bm)),6),cmm=round(float(np.mean(cm)),6),
            bmx=round(float(np.max(bm)),6),cmx=round(float(np.max(cm)),6),hr=HIT_RAD)
        self._p(f"  Baseline  hit:{r['bh']}%  miss:{r['bmm']} AU")
        self._p(f"  Corrected hit:{r['ch']}%  miss:{r['cmm']} AU")
        return r,bi,ci
    def _solve_from(self,bp,tp):
        """Solve transfer orbit from arbitrary barrel to target."""
        self._install_bodies()
        rb=np.linalg.norm(bp);vc=np.sqrt(MU/rb)
        rh=bp/rb;th=np.array([-rh[1],rh[0]]);bv=None;bm=1e30
        for sf in np.linspace(0.5,2.0,21):
            for ao in np.linspace(-np.pi,np.pi,25):
                v=vc*sf*(th*np.cos(ao)+rh*np.sin(ao))
                m=np.linalg.norm(pf(np.array([*bp,*v]),self.tf)[:2]-tp)
                if m<bm:bm=m;bv=v.copy()
        v0=bv
        for it in range(120):
            f=pf(np.array([*bp,*v0]),self.tf);miss=f[:2]-tp
            if np.linalg.norm(miss)<1e-10:return v0
            J=np.zeros((2,2));eps=1e-8
            for j in range(2):
                vp=v0.copy();vp[j]+=eps;J[:,j]=(pf(np.array([*bp,*vp]),self.tf)[:2]-f[:2])/eps
            try:v0+=np.linalg.solve(J,-miss)
            except:v0+=np.random.normal(0,0.01,2)
        return v0
    def _parallel(self,off_deg=5.0):
        """Compute parallel offset scope trajectory."""
        self._p(f"  Computing parallel scope (offset={off_deg}\u00b0)...")
        off_rad=np.radians(off_deg)
        par_angle=self.ba+off_rad
        par_bp=self.Rs*np.array([np.cos(par_angle),np.sin(par_angle)])
        par_v0=self._solve_from(par_bp,self.tp)
        par_tr=prop(np.array([*par_bp,*par_v0]),self.tf)
        # Gate states for parallel
        par_gs=[];s=np.array([*par_bp,*par_v0])
        for k in range(12):s=pf(s,self.dtg);par_gs.append(s.copy())
        # Diff vectors at each gate (parallel - primary)
        diff=[]
        for k in range(12):
            dp=par_gs[k][:2]-self.gs[k][:2]  # position diff
            dv=par_gs[k][2:]-self.gs[k][2:]   # velocity diff
            diff.append(dict(
                dx=round(float(dp[0]),5),dy=round(float(dp[1]),5),
                dvx=round(float(dv[0]),5),dvy=round(float(dv[1]),5),
                d_pos=round(float(np.linalg.norm(dp)),5),
                d_vel=round(float(np.linalg.norm(dv)),5),
                d_hdg=round(float(np.degrees(np.arctan2(par_gs[k][3],par_gs[k][2]))-
                    np.degrees(np.arctan2(self.gs[k][3],self.gs[k][2]))),2)))
        miss=np.linalg.norm(pf(np.array([*par_bp,*par_v0]),self.tf)[:2]-self.tp)
        self._p(f"    Parallel barrel: ({par_bp[0]:+.4f},{par_bp[1]:+.4f})")
        self._p(f"    Parallel miss: {miss:.2e} AU")
        par_proj=self._projectile_energy(par_bp,par_v0,
            proj=getattr(self,'par_proj',None) or getattr(self,'proj',None))
        return dict(bp=par_bp.tolist(),v0=par_v0.tolist(),
            nt=par_tr[:,0:2].tolist(),gt=[s[:2].tolist() for s in par_gs],
            diff=diff,off=off_deg,miss=round(float(miss),8),
            proj=par_proj)
    def _swarm(self,clocks=[1,3,5,9,11]):
        """Multi-barrel swarm: solve from N barrel positions to same target.
        If self.swm_positions is set, use those AU positions directly (free placement)."""
        self._p(f"  Computing swarm ({len(clocks)} barrels: {clocks})...")
        sw=[]
        free_pos=getattr(self,'swm_positions',[])
        for i,ck in enumerate(clocks):
            # Free placement override if provided and non-None
            if i<len(free_pos) and free_pos[i] is not None:
                bp=np.array(free_pos[i],dtype=float)
                ang=np.arctan2(bp[1],bp[0])
                self._p(f"    Barrel #{i+1} free @ ({bp[0]:+.4f},{bp[1]:+.4f})")
            else:
                ang=((90-ck*30)%360)*np.pi/180
                bp=self.Rs*np.array([np.cos(ang),np.sin(ang)])
            v0=self._solve_from(bp,self.tp)
            tr=prop(np.array([*bp,*v0]),self.tf)
            sgs=[];s=np.array([*bp,*v0])
            for k2 in range(12):s=pf(s,self.dtg);sgs.append(s.copy())
            miss=np.linalg.norm(pf(np.array([*bp,*v0]),self.tf)[:2]-self.tp)
            self._p(f"    Barrel @{ck}h ({np.degrees(ang):.0f}\u00b0): miss={miss:.2e}")
            # Per-barrel projectile + energy/power \u2014 default knowledge.
            sw_proj=None
            free_projs=getattr(self,'swm_projs',[])
            if i<len(free_projs) and free_projs[i] is not None:
                sw_proj=free_projs[i]
            barrel_proj=self._projectile_energy(bp,v0,proj=sw_proj)
            sw.append(dict(ck=ck,bp=bp.tolist(),v0=v0.tolist(),
                nt=tr[:,0:2].tolist(),gt=[s2[:2].tolist() for s2 in sgs],
                miss=round(float(miss),8),proj=barrel_proj))
        return sw
    def _target_orbit(self):
        """Compute target's own Keplerian orbital path (predicted + history)."""
        self._p("  Computing target orbital prediction...")
        rt=np.linalg.norm(self.tp)
        v_circ=np.sqrt(MU/rt)
        rhat=self.tp/rt;that=np.array([-rhat[1],rhat[0]])
        tgt_v0=v_circ*that
        tgt_state=np.array([*self.tp,*tgt_v0])
        # Forward path (2x flight time)
        t_prop=max(self.tf*2.0,1.0)
        tgt_tr=prop(tgt_state,t_prop)
        # Backward history path
        n_hist=max(1,round(t_prop/DT))
        h_neg=-t_prop/n_hist
        s_back=tgt_state.copy()
        hist=[s_back[:2].copy()]
        for _ in range(n_hist):
            s_back=_rk4(s_back,h_neg);hist.append(s_back[:2].copy())
        hist.reverse()
        tgt_hist=np.array(hist)
        # Target at intercept time tf
        tgt_at_tf=pf(tgt_state,self.tf)
        tgt_spd=float(np.linalg.norm(tgt_v0))
        tgt_period=2*np.pi*rt/v_circ
        # Gate-time target positions
        tgt_gates=[]
        for k in range(12):
            tg_st=pf(tgt_state,(k+1)*self.dtg)
            tgt_gates.append(tg_st[:2].tolist())
        # Relativistic data
        c_au_yr=63241.1
        beta_tgt=tgt_spd/c_au_yr
        gamma_tgt=1.0/np.sqrt(1.0-beta_tgt**2)
        self._p(f"    Target orbit: R={rt:.4f} AU, v_circ={v_circ:.4f} AU/yr, P={tgt_period:.4f} yr")
        self._p(f"    Target at T_f: ({tgt_at_tf[0]:.4f},{tgt_at_tf[1]:.4f}) AU")
        self._p(f"    Relativistic: \u03b2={beta_tgt:.2e}, \u03b3={gamma_tgt:.12f}")
        return dict(
            v0=tgt_v0.tolist(),spd=round(tgt_spd,4),
            period=round(float(tgt_period),4),
            r=round(float(rt),4),
            at_tf=tgt_at_tf[:2].tolist(),
            path=tgt_tr[:,0:2].tolist(),
            history=tgt_hist.tolist(),
            gates=tgt_gates,
            spd_kms=round(tgt_spd*1.496e8/3.156e7,2),
            beta=round(float(beta_tgt),12),
            gamma=round(float(gamma_tgt),12))
    def _projectile_energy(self,bp,v0,proj=None,dv_gates=None):
        """Energy budget for one barrel firing a given projectile.

        Every barrel ships with a default projectile spec (diameter_m, density_kg_m3)
        so 'power needed to make the hit' is always available. When the target
        changes, |v0| changes and the energy/power recompute automatically.
        """
        if proj is None:proj=dict(DEFAULT_PROJECTILE)
        # --- geometry ---
        d=float(proj.get('diameter_m',100.0))
        rho=float(proj.get('density_kg_m3',2600.0))  # rocky asteroid default
        vol=np.pi*d**3/6.0
        m=proj.get('mass_kg')
        m=float(m) if m is not None else vol*rho
        # --- unit conversions AU/yr -> SI ---
        AU_M=1.496e11;YR_S=3.156e7
        v0m=float(np.linalg.norm(v0))          # AU/yr
        v0_ms=v0m*AU_M/YR_S                    # m/s
        # --- kinetic energy at launch ---
        ke_j=0.5*m*v0_ms*v0_ms
        # --- gravitational potential energy change: bp radius -> target radius ---
        rb=float(np.linalg.norm(bp));rt=float(np.linalg.norm(self.tp))
        GM_SI=MU*(AU_M**3)/(YR_S**2)           # m^3/s^2 (heliocentric)
        gpe=GM_SI*m*(1.0/max(rb,1e-6)-1.0/max(rt,1e-6))/AU_M  # signed J
        total_j=ke_j+max(gpe,0.0)
        # --- Δv budget: launch + sum of gate correction magnitudes ---
        dv_g=float(np.sum(dv_gates)) if dv_gates is not None else 0.0
        dv_total=v0m+dv_g                       # AU/yr
        dv_total_ms=dv_total*AU_M/YR_S
        # --- Tsiolkovsky fuel mass fraction (Isp in seconds; ve = Isp*g0) ---
        Isp=float(proj.get('Isp_s',350.0));g0=9.80665
        ve=Isp*g0
        mf_ratio=float(np.exp(dv_total_ms/ve))  # m0/mf
        fuel_frac=1.0-1.0/mf_ratio
        # --- power: if total energy is delivered across the flight time ---
        tf_s=float(self.tf)*YR_S
        p_avg_w=total_j/tf_s if tf_s>0 else 0.0
        # --- TNT equivalence + magnitudes the eye can read ---
        tnt_kt=ke_j/4.184e12                    # kilotons TNT
        return dict(
            diameter_m=round(d,4),
            mass_kg=float(f'{m:.4e}'),
            density_kg_m3=round(rho,2),
            v0_kms=round(v0_ms/1000.0,3),
            dv_launch_kms=round(v0_ms/1000.0,3),
            dv_gates_kms=round(dv_g*AU_M/YR_S/1000.0,4),
            dv_total_kms=round(dv_total_ms/1000.0,3),
            ke_j=float(f'{ke_j:.4e}'),
            gpe_j=float(f'{gpe:.4e}'),
            total_j=float(f'{total_j:.4e}'),
            tnt_kt=round(tnt_kt,4),
            power_avg_W=float(f'{p_avg_w:.4e}'),
            Isp_s=Isp,
            exhaust_ve_ms=round(ve,2),
            mass_ratio=round(mf_ratio,4),
            fuel_frac=round(fuel_frac,5))
    def _energy_model(self):
        """Compute energy requirements based on mass, distance, velocity."""
        self._p("  Computing energy/mass model...")
        v0m=np.linalg.norm(self.v0)
        rb=np.linalg.norm(self.bp);rt=np.linalg.norm(self.tp)
        dist=np.linalg.norm(self.tp-self.bp)
        # Reference masses (kg) — shown for scale; the per-barrel projectile
        # spec (proj_default) is what actually drives the live energy/power budget.
        masses={'1km asteroid':5.2e11,'10km comet':4.2e12,'100m bolide':2.6e9,
                '1m probe':1000.0,'100kg interceptor':100.0}
        energy={}
        for label,m in masses.items():
            ke=0.5*m*(v0m*1.496e11/3.156e7)**2  # KE in Joules (v in m/s)
            # Gravitational potential energy to escape from barrel radius
            gpe=MU*(1.496e11)**3/(3.156e7)**2*m/rb/1.496e11  # approximate
            energy[label]=dict(mass_kg=m,ke_j=float(f'{ke:.3e}'),
                total_j=float(f'{(ke+gpe):.3e}'),
                ke_tnt=round(ke/4.184e9,2))  # kilotons TNT equivalent
        # Primary-barrel projectile + power need (what the user actually flies)
        prim_proj=self._projectile_energy(self.bp,self.v0,
            proj=getattr(self,'proj',None))
        # Δv budget
        dv_launch=round(float(v0m),4)
        # Relativistic corrections
        c_au_yr=63241.1;v0_kms=v0m*1.496e8/3.156e7
        beta_proj=v0m/c_au_yr
        gamma_proj=1.0/np.sqrt(1.0-beta_proj**2)
        time_dil=1.0/gamma_proj  # proper time factor
        # Galactic/environmental parameters
        galactic=dict(
            solar_v_galactic_kms=230.0,
            solar_v_lsr_kms=20.0,
            local_ism_density_cm3=0.1,
            solar_luminosity_W=3.828e26,
            radiation_pressure_au1_Nm2=4.56e-6)
        self._p(f"    |v0|={v0m:.4f} AU/yr ({v0_kms:.2f} km/s)")
        self._p(f"    Barrel-Target dist: {dist:.4f} AU ({dist*1.496e8:.0f} km)")
        self._p(f"    Energy (1km asteroid): {energy['1km asteroid']['ke_j']:.2e} J")
        self._p(f"    Projectile \u03b2={beta_proj:.2e}, \u03b3={gamma_proj:.12f}")
        return dict(
            v0_mag=round(float(v0m),4),
            v0_kms=round(v0_kms,2),
            barrel_r=round(float(rb),4),target_r=round(float(rt),4),
            dist_au=round(float(dist),4),
            dist_km=round(float(dist*1.496e8),0),
            dv_launch=dv_launch,
            beta=round(float(beta_proj),12),
            gamma=round(float(gamma_proj),12),
            time_dilation=round(float(time_dil),12),
            c_au_yr=c_au_yr,
            energy=energy,galactic=galactic,
            proj_default=dict(DEFAULT_PROJECTILE),
            primary_proj=prim_proj)
    def viz(self,r,bi,ci):
        nom=prop(np.array([*self.bp,*self.v0]),self.tf)
        gt=np.array([s[:2] for s in self.gs])
        gi=[]
        prev=np.array([*self.bp,*self.v0])  # barrel state as "gate 0"
        for k in range(12):
            gs=self.gs[k];rsun=float(np.linalg.norm(gs[:2]));spd=float(np.linalg.norm(gs[2:]))
            grav=MU/rsun**2;Jv=self.gJ[k][:,2:]
            try:jc=float(np.linalg.cond(Jv))
            except:jc=9999
            dp=gs[:2]-prev[:2];dv=gs[2:]-prev[2:]
            arc=float(np.linalg.norm(dp))
            spd_prev=float(np.linalg.norm(prev[2:]))
            dspd=spd-spd_prev
            hdg=float(np.degrees(np.arctan2(gs[3],gs[2])))
            hdg_prev=float(np.degrees(np.arctan2(prev[3],prev[2])))
            dhdg=hdg-hdg_prev
            while dhdg>180:dhdg-=360
            while dhdg<-180:dhdg+=360
            curv=abs(dhdg*np.pi/180)/arc if arc>1e-10 else 0
            gi.append(dict(n=k+1,pos=[round(gs[0],4),round(gs[1],4)],
                vx=round(float(gs[2]),4),vy=round(float(gs[3]),4),
                spd=round(spd,3),rsun=round(rsun,3),grav=round(grav,2),
                tg=round((k+1)*self.dtg,4),tr=round(self.tf-(k+1)*self.dtg,4),
                jc=round(min(jc,9999),1),arc=round(arc,4),dspd=round(dspd,3),
                hdg=round(hdg,1),dhdg=round(dhdg,1),curv=round(curv,2)))
            prev=gs.copy()
        # Parallel scope + swarm
        par=self._parallel(5.0)
        swm_clks=getattr(self,'swm_clocks',[1,3,5,9,11])
        swm=self._swarm(swm_clks)
        # Target orbital prediction + energy model
        tgt_orb=self._target_orbit()
        ener=self._energy_model()
        return dict(B=self.bp.tolist(),TG=self.tp.tolist(),gxy=self.gxy.tolist(),
            gt=gt.tolist(),nt=nom[:,0:2].tolist(),Rs=self.Rs,Rs_base=self.Rs_base,hr=HIT_RAD,tf=self.tf,
            dtg=self.dtg,T=self.T.tolist(),v0=self.v0.tolist(),res=r,bi=bi.tolist(),
            ci=ci.tolist(),log=self.log,gi=gi,par=par,swm=swm,
            tgt_orb=tgt_orb,ener=ener,
            sun_pos=self.sun_pos.tolist(),
            sun_z=float(self.sun_z),
            tz=float(self.tz),
            use_perturbations=bool(self.use_perturbations),
            solar=_serialise_solar(self.solar))

HTML=r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>TENSOR-FLOWER Comet Redirection System v5.1</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#c8c0b0;font-family:'Segoe UI',sans-serif;overflow:hidden;height:100vh;display:flex;flex-direction:column}
#hdr{background:linear-gradient(180deg,#1a1610,#0e0c08);border-bottom:2px solid #5a4a32;padding:8px 16px;display:flex;align-items:center;gap:16px;flex-shrink:0}
#hdr h1{font:bold 14px 'Times New Roman',serif;color:#c8a96e;letter-spacing:2px}
.hs{font-size:11px;color:#889}.hv{color:#ffd700;font-weight:700}
#tabs{display:flex;background:#151210;border-bottom:2px solid #5a4a32;flex-shrink:0}
.tb{flex:1;padding:10px 0;text-align:center;cursor:pointer;font:bold 11px 'Segoe UI',sans-serif;color:#887;letter-spacing:1px;border-right:1px solid #2a2418;transition:all .15s;user-select:none}
.tb:last-child{border-right:none}.tb:hover{background:#1e1a14;color:#c8a96e}
.tb.on{background:#2a2418;color:#ffd700;box-shadow:inset 0 -3px 0 #c8a96e}
#ct{flex:1;position:relative;overflow:hidden}
.pg{position:absolute;inset:0;display:none;overflow:auto}.pg.on{display:flex}
canvas{display:block}
#imp-pg{gap:4px;padding:4px}#imp-pg canvas{flex:1;border:1px solid #2a2418;border-radius:4px}
#st-pg{flex-direction:column;padding:20px 40px;gap:16px;align-items:center}
#st-pg .cd{max-width:1280px}
#st-pg>div:first-child{width:100%;max-width:1280px}
.cd{background:linear-gradient(180deg,#1a1610,#12100c);border:1px solid #3a3020;border-radius:8px;padding:16px 24px;width:100%;max-width:700px}
.cd h2{font:bold 13px 'Times New Roman',serif;color:#c8a96e;margin-bottom:10px;letter-spacing:1px;border-bottom:1px solid #2a2418;padding-bottom:6px}
.rw{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid rgba(90,74,50,.12)}
.rw .lb{color:#889}.rw .vl{color:#ffd700;font-weight:700}
.bw{height:16px;background:#0e0c08;border:1px solid #3a3020;border-radius:4px;overflow:hidden;margin:3px 0}
.bf{height:100%;border-radius:3px;transition:width .5s}
.tm{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-top:8px}
.tc{text-align:center;padding:7px 3px;font:bold 11px monospace;border-radius:4px;border:1px solid #2a2418}
#con-pg{padding:0}#con-pg pre{flex:1;padding:12px;font:12px Consolas,monospace;color:#44ff88;background:#060810;overflow:auto;white-space:pre-wrap;line-height:1.6}
#tip{position:fixed;z-index:999;display:none;background:linear-gradient(180deg,rgba(30,24,16,.97),rgba(20,16,10,.98));border:2px solid #c8a96e;border-radius:6px;padding:8px 12px;pointer-events:none;font-size:11px;max-width:260px;box-shadow:0 4px 16px rgba(0,0,0,.7);line-height:1.5}
#tip b{color:#ffd700;font-size:12px}#tip .tl{color:#889}#tip .tv{color:#ffdd44;font-weight:700}
#esc{position:fixed;inset:0;z-index:900;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center}
#esc.on{display:flex}
#ep{background:linear-gradient(180deg,#1e1a14,#12100c);border:2px solid #c8a96e;border-radius:10px;padding:24px 32px;max-width:520px;width:90%;max-height:85vh;overflow-y:auto;box-shadow:0 8px 40px rgba(0,0,0,.6)}
#ep h2{font:bold 16px 'Times New Roman',serif;color:#c8a96e;letter-spacing:2px;margin-bottom:12px;text-align:center}
#ep h3{font:bold 11px sans-serif;color:#ffd700;letter-spacing:1px;margin:12px 0 6px;border-bottom:1px solid #3a3020;padding-bottom:4px}
.er{padding:3px 0;font-size:11px;color:#aaa;display:flex;gap:8px}
.ek{background:#2a2418;border:1px solid #5a4a32;border-radius:3px;padding:1px 6px;font:bold 10px monospace;color:#c8a96e;white-space:nowrap}
.ec{text-align:center;font-size:10px;color:#556;margin-top:16px}
#ctrl{background:#12100c;border-top:2px solid #5a4a32;padding:6px 16px;display:flex;align-items:center;gap:10px;flex-shrink:0}
#ctrl button{background:#2a2418;border:1px solid #5a4a32;color:#c8a96e;padding:4px 12px;font:bold 10px monospace;cursor:pointer;border-radius:3px}
#ctrl button:hover{background:#3a3020;color:#ffd700}
#ctrl button.act{background:#5a4a32;color:#ffd700}
#timeline{flex:1;-webkit-appearance:none;height:6px;background:#2a2418;border-radius:3px;outline:none}
#timeline::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:#ffd700;border-radius:50%;cursor:pointer}
.clbl{font:bold 10px monospace;color:#889;white-space:nowrap}
.clbl b{color:#ffd700}
#hud{position:absolute;top:48px;right:12px;background:rgba(10,10,26,.88);border:1px solid #5a4a32;border-radius:6px;padding:6px 10px;font:10px monospace;color:#c8a96e;pointer-events:none;z-index:10;line-height:1.6;min-width:180px}
#hud .hl{color:#ffd700;font-weight:bold}#hud .hd{color:#889}
#coord{position:absolute;bottom:6px;left:50%;transform:translateX(-50%);background:rgba(10,10,26,.85);border:1px solid #3a3020;border-radius:4px;padding:3px 10px;font:bold 10px monospace;color:#c8a96e;pointer-events:none;z-index:10}
#tpan{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:800;display:none;background:linear-gradient(180deg,#1e1a14,#12100c);border:2px solid #c8a96e;border-radius:10px;padding:20px 28px;min-width:320px;box-shadow:0 8px 40px rgba(0,0,0,.7)}
#tpan.on{display:block}
#tpan h3{font:bold 13px 'Times New Roman',serif;color:#c8a96e;margin-bottom:10px;letter-spacing:1px}
#tpan label{font:11px sans-serif;color:#889;display:flex;align-items:center;gap:8px;margin:6px 0}
#tpan input{background:#0a0a1a;border:1px solid #5a4a32;color:#ffd700;padding:4px 8px;font:bold 12px monospace;width:100px;border-radius:3px;text-align:right}
#tpan .tdist{font:bold 12px monospace;color:#44ff88;margin-top:8px}
#tpan .tnote{font:9px sans-serif;color:#556;margin-top:10px}
#tpan button{background:#2a2418;border:1px solid #5a4a32;color:#c8a96e;padding:5px 14px;font:bold 10px monospace;cursor:pointer;border-radius:3px;margin-top:8px}
#tpan button:hover{background:#3a3020;color:#ffd700}
.cfg-in{background:#1a1610;border:1px solid #5a4a32;color:#ffdd44;padding:3px 6px;font:11px monospace;border-radius:3px;width:100px;text-align:right}
.cfg-in:focus{border-color:#c8a96e;outline:none;box-shadow:0 0 4px rgba(200,169,110,.3)}
select.cfg-in{width:140px;text-align:left;cursor:pointer}
#dsp{position:fixed;top:80px;left:10px;z-index:800;display:none;background:rgba(14,12,8,.96);border:2px solid #5a4a32;border-radius:8px;padding:10px 14px;min-width:220px;max-height:80vh;overflow-y:auto;box-shadow:0 6px 24px rgba(0,0,0,.7)}
#dsp.on{display:block}
/* Shared clickable LEGEND panel — always visible, acts as both legend and layer switch */
#legend{position:fixed;bottom:52px;left:10px;z-index:850;background:#0e0c08;border:2px solid #5a4a32;border-radius:8px;padding:8px 10px;min-width:210px;max-width:260px;max-height:60vh;overflow-y:auto;box-shadow:0 6px 22px rgba(0,0,0,.85),0 0 0 3px rgba(0,0,0,.55);font:10px sans-serif;color:#c8a96e;backdrop-filter:blur(4px)}
#legend.collapsed{padding:4px 8px;min-width:0}
#legend.collapsed .lgrow{display:none}
#legend h4{font:bold 10px sans-serif;color:#c8a96e;margin:0 0 4px;padding-bottom:3px;border-bottom:1px solid #3a3020;letter-spacing:1px;display:flex;align-items:center;justify-content:space-between;gap:6px;cursor:pointer;user-select:none}
#legend h4 .tog{color:#ffd700;font-family:monospace}
.lgrow{display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;user-select:none}
.lgrow:hover{color:#ffd700}
.lgrow input[type=checkbox]{accent-color:#c8a96e;cursor:pointer;margin:0}
.lgsw{width:12px;height:12px;border-radius:2px;border:1px solid rgba(255,255,255,.25);flex-shrink:0}
.lgtxt{font-size:10px;flex:1}
#dsp h4{font:bold 11px sans-serif;color:#c8a96e;margin:0 0 6px;border-bottom:1px solid #3a3020;padding-bottom:4px;letter-spacing:1px}
#dsp label{font:10px sans-serif;color:#889;display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;user-select:none}
#dsp label:hover{color:#c8a96e}
#dsp input[type=checkbox]{accent-color:#c8a96e;cursor:pointer}
#dsp input[type=range]{width:100%;accent-color:#c8a96e;height:4px}
.dsp-val{color:#ffd700;font:bold 10px monospace;min-width:30px;text-align:right}
.cfg-btn{padding:8px 20px;font:bold 11px monospace;border:2px solid;border-radius:5px;cursor:pointer;transition:all .2s}
.cfg-btn:hover{filter:brightness(1.3);transform:scale(1.02)}
#cfg-pg .cd{margin-bottom:12px}
/* ─── LOADING SPLASH ─────────────────────────────────────────────────── */
/* Shown immediately on page load while the embedded JS parses the viz
   payload, builds XS, wires handlers, and renders the first frame. The JS
   dismisses it once the first draw() has completed.                      */
#load{position:fixed;inset:0;z-index:9999;background:radial-gradient(circle at 50% 40%,#141c28 0%,#040812 70%,#000 100%);display:flex;flex-direction:column;align-items:center;justify-content:center;color:#c8a96e;font:14px 'Segoe UI',sans-serif;transition:opacity .35s ease-out}
#load.done{opacity:0;pointer-events:none}
#load .ttl{font:bold 22px 'Times New Roman',serif;letter-spacing:3px;color:#ffd700;text-shadow:0 0 18px rgba(255,215,0,.55);margin-bottom:4px}
#load .sub{font:11px 'Consolas',monospace;color:#889;letter-spacing:1px;margin-bottom:22px}
#load .ring{width:72px;height:72px;border:4px solid rgba(200,169,110,.2);border-top-color:#ffd700;border-right-color:#c8a96e;border-radius:50%;animation:ldsp .9s linear infinite;margin-bottom:20px;box-shadow:0 0 30px rgba(255,215,0,.25) inset}
@keyframes ldsp{to{transform:rotate(360deg)}}
#load .bar{width:440px;max-width:80vw;height:8px;background:rgba(30,30,40,.8);border:1px solid #3a3020;border-radius:4px;overflow:hidden;margin-bottom:10px}
#load .bar .fill{height:100%;width:0%;background:linear-gradient(90deg,#c8a96e,#ffd700,#c8a96e);background-size:200% 100%;animation:ldfill 1.6s linear infinite;transition:width .18s ease-out}
@keyframes ldfill{0%{background-position:0% 50%}100%{background-position:200% 50%}}
#load .steps{width:440px;max-width:80vw;font:11px 'Consolas',monospace;color:#667;line-height:1.6}
#load .steps .s{display:flex;justify-content:space-between;padding:1px 4px}
#load .steps .s.ok{color:#44ff88}
#load .steps .s.cur{color:#ffd700}
#load .pct{font:bold 14px 'Consolas',monospace;color:#ffd700;margin-bottom:8px;letter-spacing:1px}
</style></head><body>
<!-- Loading splash — dismissed by the JS at the end of its init pass -->
<div id="load">
<div class="ttl">TENSOR-FLOWER</div>
<div class="sub">Comet Redirection System — initializing</div>
<div class="ring"></div>
<div class="pct" id="ldPct">0%</div>
<div class="bar"><div class="fill" id="ldFill"></div></div>
<div class="steps" id="ldSteps">
<div class="s" id="ls1">◇ Parsing viz payload (B, TG, gates, solar) …<span>·</span></div>
<div class="s" id="ls2">◇ Building Flower-of-Life lattice + XS intersections …<span>·</span></div>
<div class="s" id="ls3">◇ Resolving Fibonacci spiral indices …<span>·</span></div>
<div class="s" id="ls4">◇ Wiring hover / click / legend handlers …<span>·</span></div>
<div class="s" id="ls5">◇ Rendering first SCOPE frame …<span>·</span></div>
</div>
</div>
<div id="hdr">
<h1>TENSOR-FLOWER COMET REDIRECTION <span id="hScope" style="font:bold 10px 'Consolas',monospace;color:#c8a96e;letter-spacing:1px;margin-left:8px"></span></h1>
<span class="hs">Baseline:<span class="hv" id="hbh"></span></span>
<span class="hs">Corrected:<span class="hv" id="hch"></span></span>
<span class="hs">Improve:<span class="hv" id="him"></span></span>
<span class="hs">v<sub>0</sub>:<span class="hv" id="hv0"></span></span>
<span class="hs">Flight:<span class="hv" id="hfl"></span></span>
<span class="hs">Miss:<span class="hv" id="hms"></span></span>
<span class="hs">Gates:<span class="hv">12</span></span>
<span class="hs" style="margin-left:auto;color:#556">SPACE=Play &middot; T=Target &middot; P=Parallel &middot; S=Swarm &middot; ESC=Info</span>
</div>
<div id="tabs">
<div class="tb on" data-t="sc-pg">1 SCOPE</div>
<div class="tb" data-t="imp-pg">2 IMPACT</div>
<div class="tb" data-t="v3-pg">3 3D VIEW</div>
<div class="tb" data-t="st-pg">4 STATS</div>
<div class="tb" data-t="con-pg">5 CONSOLE</div>
<div class="tb" data-t="cfg-pg">6 CONFIG</div>
<div class="tb" data-t="sal-pg">7 SALVO</div>
<div class="tb" data-t="trj-pg">8 TRAJECTORY</div>
<div class="tb" data-t="sph-pg">9 SPHERE NAV</div>
<div class="tb" data-t="s3d-pg">10 3D SPHERE</div>
<div class="tb" data-t="par-pg">11 PARALLEL</div>
<div class="tb" data-t="p3d-pg">12 PARALLEL 3D</div>
</div>
<div id="ct">
<div class="pg on" id="sc-pg"><canvas id="sC"></canvas><div id="hud"></div><div id="coord"></div></div>
<div class="pg" id="imp-pg"><canvas id="bC"></canvas><canvas id="cC"></canvas></div>
<div class="pg" id="v3-pg"><canvas id="v3C"></canvas><div id="v3Tip" style="position:absolute;display:none;background:rgba(10,10,26,.95);border:2px solid #c8a96e;border-radius:6px;padding:8px 12px;font:11px monospace;color:#c8a96e;pointer-events:none;z-index:50;max-width:300px;box-shadow:0 4px 16px rgba(0,0,0,.7)"></div><div id="v3Hud" style="position:absolute;top:8px;right:8px;background:rgba(10,10,26,.92);border:1px solid #5a4a32;border-radius:6px;padding:10px;font:10px monospace;color:#c8a96e;max-width:260px;pointer-events:none"></div><div id="v3Coord" style="position:absolute;bottom:56px;left:50%;transform:translateX(-50%);background:rgba(10,10,26,.88);border:1px solid #5a4a32;border-radius:4px;padding:4px 14px;font:bold 11px monospace;color:#ffd700;pointer-events:none;white-space:nowrap"></div></div>
<div class="pg" id="st-pg"></div>
<div class="pg" id="con-pg"><pre id="clog"></pre></div>
<div class="pg" id="cfg-pg" style="overflow-y:auto;padding:16px"><div id="cfgContent"></div></div>
<div class="pg" id="sal-pg"><canvas id="salC"></canvas><div id="salInfo" style="position:absolute;top:8px;right:8px;background:rgba(10,10,26,.9);border:1px solid #5a4a32;border-radius:5px;padding:10px;font:10px monospace;color:#c8a96e;max-width:280px;overflow-y:auto;max-height:90%;pointer-events:none"></div><div id="salLive" style="position:absolute;bottom:12px;left:12px;background:rgba(10,10,26,.92);border:2px solid #c8a96e;border-radius:6px;padding:10px 14px;font:10px monospace;color:#c8a96e;max-width:420px;max-height:40%;overflow-y:auto;pointer-events:none;box-shadow:0 2px 12px rgba(0,0,0,.6)"></div><div id="salTip" style="position:absolute;display:none;background:rgba(20,16,10,.95);border:2px solid #ffd700;border-radius:6px;padding:8px 12px;font:11px monospace;color:#c8a96e;pointer-events:none;z-index:50;max-width:320px;box-shadow:0 4px 16px rgba(0,0,0,.7)"></div></div>
<div class="pg" id="sph-pg"><canvas id="sphC"></canvas><div id="sphCoord" style="position:absolute;bottom:8px;left:50%;transform:translateX(-50%);background:rgba(10,10,26,.9);border:2px solid #c8a96e;border-radius:5px;padding:6px 16px;font:bold 12px monospace;color:#ffd700;pointer-events:none;z-index:10;white-space:nowrap"></div><div id="sphInfo" style="position:absolute;top:8px;right:8px;background:rgba(10,10,26,.92);border:1px solid #5a4a32;border-radius:6px;padding:10px;font:10px monospace;color:#c8a96e;max-width:320px;max-height:90%;overflow-y:auto;pointer-events:none"></div></div>
<div class="pg" id="s3d-pg"><canvas id="s3dC"></canvas><div id="s3dHud" style="position:absolute;top:8px;right:8px;background:rgba(10,10,26,.92);border:1px solid #5a4a32;border-radius:6px;padding:10px;font:10px monospace;color:#c8a96e;max-width:280px;pointer-events:none"></div><div id="s3dTip" style="position:absolute;display:none;background:rgba(10,10,26,.95);border:2px solid #c8a96e;border-radius:6px;padding:8px 12px;font:11px monospace;color:#c8a96e;pointer-events:none;z-index:50;max-width:300px;box-shadow:0 4px 16px rgba(0,0,0,.7)"></div><div id="s3dCoord" style="position:absolute;bottom:56px;left:50%;transform:translateX(-50%);background:rgba(10,10,26,.88);border:1px solid #5a4a32;border-radius:4px;padding:4px 14px;font:bold 11px monospace;color:#ffd700;pointer-events:none;white-space:nowrap"></div></div>
<div class="pg" id="trj-pg"><canvas id="trjC"></canvas><div id="trjTip" style="position:absolute;display:none;background:rgba(10,10,26,.95);border:2px solid #ffd700;border-radius:6px;padding:8px 12px;font:11px monospace;color:#c8a96e;pointer-events:none;z-index:50;max-width:340px;box-shadow:0 4px 16px rgba(0,0,0,.7)"></div><div id="trjHud" style="position:absolute;top:12px;left:12px;background:linear-gradient(180deg,rgba(30,24,16,.95),rgba(20,16,10,.92));border:2px solid #5a4a32;border-radius:6px;padding:10px 14px;font:10px monospace;color:#c8a96e;max-width:260px;pointer-events:none;box-shadow:0 2px 12px rgba(0,0,0,.6)"></div><div id="trjPicker" style="position:absolute;top:12px;right:12px;max-width:46%;z-index:20;display:flex;flex-wrap:wrap;gap:4px;justify-content:flex-end"></div><div id="trjInfo" style="position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:linear-gradient(180deg,rgba(30,24,16,.92),rgba(20,16,10,.95));border:2px solid #5a4a32;border-radius:6px;padding:6px 20px;font:10px monospace;color:#c8a96e;pointer-events:none"></div></div>
<div class="pg" id="par-pg"><canvas id="parC"></canvas><div id="parTip" style="position:fixed;display:none;background:rgba(10,26,18,.95);border:2px solid #00dd88;border-radius:6px;padding:8px 12px;font:11px monospace;color:#c8f5d8;pointer-events:none;z-index:999;max-width:330px;box-shadow:0 4px 16px rgba(0,0,0,.7);line-height:1.55"></div><div id="parHud" style="position:absolute;top:48px;right:12px;background:rgba(10,10,26,.92);border:1px solid #00cc88;border-radius:6px;padding:8px 12px;font:10px monospace;color:#c8f5d8;pointer-events:none;z-index:10;line-height:1.55;min-width:240px;max-height:88%;overflow-y:auto"></div><div id="parCoord" style="position:absolute;bottom:6px;left:50%;transform:translateX(-50%);background:rgba(10,10,26,.88);border:1px solid #00aa66;border-radius:4px;padding:3px 10px;font:bold 10px monospace;color:#44ffaa;pointer-events:none;z-index:10"></div></div>
<div class="pg" id="p3d-pg"><canvas id="p3dC"></canvas><div id="p3dTip" style="position:absolute;display:none;background:rgba(10,10,26,.95);border:2px solid #00dd88;border-radius:6px;padding:8px 12px;font:11px monospace;color:#c8f5d8;pointer-events:none;z-index:50;max-width:300px;box-shadow:0 4px 16px rgba(0,0,0,.7)"></div><div id="p3dHud" style="position:absolute;top:8px;right:8px;background:rgba(10,10,26,.92);border:1px solid #00cc88;border-radius:6px;padding:10px;font:10px monospace;color:#c8f5d8;max-width:280px;pointer-events:none"></div><div id="p3dCoord" style="position:absolute;bottom:56px;left:50%;transform:translateX(-50%);background:rgba(10,10,26,.88);border:1px solid #00aa66;border-radius:4px;padding:4px 14px;font:bold 11px monospace;color:#44ffaa;pointer-events:none;white-space:nowrap"></div></div>
</div>
<div id="ctrl">
<button id="playBtn">▶ PLAY</button>
<button id="stepBk">◀◀</button>
<button id="stepFw">▶▶</button>
<input type="range" id="timeline" min="0" max="1000" value="0">
<span class="clbl">T+<b id="tDisp">0.000</b>yr</span>
<span class="clbl">Spd:<b id="sDisp">0.00</b></span>
<span class="clbl">Gate:<b id="gDisp">--</b></span>
<button id="tgtBtn">T: TARGET</button>
<span style="border-left:1px solid #5a4a32;margin:0 4px"></span>
<button id="hkZoomIn" title="Zoom In (+)">&#43; Zoom</button>
<button id="hkZoomOut" title="Zoom Out (-)">&#8722; Zoom</button>
<button id="hkReset" title="Reset Camera (R)">R Reset</button>
<button id="hkPar" title="Toggle Parallel (P)">P Par</button>
<button id="hkSwm" title="Toggle Swarm (S)">S Swm</button>
<button id="hkEsc" title="Info Panel (ESC)">? Info</button>
<button id="hkDsp" title="Display Settings">&#9881; Display</button>
</div>
<div id="tpan">
<h3>◇ TARGET LOCATION</h3>
<label>X (AU): <input id="tgtX" type="number" step="0.01"></label>
<label>Y (AU): <input id="tgtY" type="number" step="0.01"></label>
<div class="tdist">Distance from FTOP: <span id="tgtD">0</span> AU</div>
<button id="tgtClose">CLOSE (T)</button>
<div class="tnote">Note: Changing target requires re-simulation. Click scope to measure AU distance.</div>
</div>
<div id="tip"></div>
<div id="dsp">
<h4>&#9881; DISPLAY SETTINGS</h4>
<div style="margin-bottom:8px">
<label>Font Size: <span class="dsp-val" id="fsVal">1.0x</span></label>
<input type="range" id="fsSlider" min="20" max="150" value="100" step="5">
</div>
<h4>VISIBLE LAYERS</h4>
<label><input type="checkbox" id="dL" checked>Labels &amp; Text</label>
<label><input type="checkbox" id="dG" checked>Gate Markers</label>
<label><input type="checkbox" id="dO" checked>Reference Orbits</label>
<label><input type="checkbox" id="dT" checked>Trajectory Path</label>
<label><input type="checkbox" id="dX" checked>Cross-sections (Fibonacci)</label>
<label><input type="checkbox" id="dTgt" checked>Target &amp; Target Orbit</label>
<label><input type="checkbox" id="dBar" checked>Barrel</label>
<label><input type="checkbox" id="dPar" checked>Parallel Scope</label>
<label><input type="checkbox" id="dSwm" checked>Swarm Barrels</label>
<label><input type="checkbox" id="dSph" checked>Sphere Addresses</label>
<label><input type="checkbox" id="dEn" checked>Energy / Physics</label>
<label><input type="checkbox" id="dGrid" checked>Grid Lines</label>
<div style="margin-top:8px;display:flex;gap:6px">
<button onclick="dspAll(true)" style="font:bold 9px monospace;background:#2a2418;border:1px solid #5a4a32;color:#c8a96e;padding:3px 8px;border-radius:3px;cursor:pointer">ALL ON</button>
<button onclick="dspAll(false)" style="font:bold 9px monospace;background:#2a2418;border:1px solid #5a4a32;color:#c8a96e;padding:3px 8px;border-radius:3px;cursor:pointer">ALL OFF</button>
</div>
</div>
<!-- Shared clickable LEGEND — replaces canvas-drawn legends; checkboxes toggle
     the same DV layer flags and showParallel/showSwarm booleans, then redraw. -->
<div id="legend"><h4><span>◆ LEGEND &amp; LAYERS</span><span class="tog" id="lgTog">[−]</span></h4>
<div id="lgBody"></div></div>
<div id="esc"><div id="ep">
<h2>TENSOR-FLOWER COMET REDIRECTION SYSTEM</h2>
<p style="font-size:10px;color:#889;text-align:center;margin-bottom:10px">Hit.py v5.16 &mdash; Tri-axis 360&deg; sphere coords (Z&phi; Y&phi; X&phi;) &middot; opaque legend (no bleed) &middot; scope-distance on tab 1 title</p>
<h3>PROGRAM OVERVIEW</h3>
<div class="er" style="color:#ddd">This system simulates redirecting a comet from a barrel (pusher) position through a gravitational field to a target position using 12 mid-course correction gates arranged in a Flower-of-Life lattice. The physics uses 2-body Keplerian gravity with RK4 numerical integration. A Newton shooting algorithm solves for the initial velocity, and precomputed Jacobians at each gate enable real-time trajectory corrections during Monte Carlo simulations.</div>
<h3 style="color:#ffd700">SCOPE AS STENCIL SCALE-UNIT (auto-grow)</h3>
<div class="er" style="color:#ddd">Hit.py is a renderer of a single <b>stencil template</b>: a scope-scale unit of measure. The template is Flower-of-Life, always the same 31-circle geometry, always the same 12 clock-face gates &mdash; the <i>only</i> thing that ever changes between runs is its <b>diameter</b>. More AU to cover? The same stencil gets bigger. No new spheres are ever added, no tiling happens; the same lattice is simply <i>scrolled out</i> until the whole scene fits inside it.</div>
<div class="er" style="color:#ffd700;border:1px solid #ffd700;padding:6px;border-radius:4px;margin-top:6px">
<b>Rule (enforced on every /resim):</b> if the target <i>or</i> any placed barrel (primary, parallel, swarm) sits outside the current scope ring, the scope radius R<sub>s</sub> is rescaled uniformly to contain it &mdash; <i>not</i> by adding new rings or tiles, but by making the one existing template larger. Conversely, if everything is inside the base scope, R<sub>s</sub> shrinks back down to R<sub>base</sub>. The on-screen FoL lattice, gates, reference orbits and labels all occupy the <b>same fraction of the viewport regardless of R<sub>s</sub></b> &mdash; the canvas base-scale is normalised by R<sub>base</sub>/R<sub>s</sub>, so a 12-AU scope looks identical to a 1.2-AU scope, just covering 10&times; more AU per pixel.
</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">r_need = max(|T|, |B|, max |B<sub>swarm</sub>|, 1.52)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">R<sub>s</sub> = max(R<sub>base</sub>, R<sub>base</sub> &times; &lceil;(r_need &times; 1.10) / R<sub>base</sub>&rceil;)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">baseSc<sub>pix</sub> = min(w,h)/5.6 &times; (R<sub>base</sub> / R<sub>s</sub>)   &larr; applied in tabs 1, 3, 7, 8, 11, 12</div>
<div class="er" style="color:#ddd"><b>Why this matters:</b> the "stencil-is-a-scale-unit" idea means the same template is valid at <i>any</i> scale &mdash; solar-system, interstellar, galactic, intergalactic. When you set a target 8 AU away, the scope ring, gate positions, FoL lattice, tensor mapping lines, and cross-section intersections all resize by the same factor. The geometry of the template (31 circles, 12 gates, ε = Rs&sdot;0.58 inner, ΔR = Rs between rings) is <b>invariant</b> under that rescale; only the AU value under each unit changes. The title-bar readout (R<sub>s</sub> AU &middot; ⌀ 2R<sub>s</sub> AU &middot; N&times; base) makes the current scale-unit explicit at all times.</div>
<div class="er" style="color:#ddd"><b>Propagation:</b> every tab honours the grown R<sub>s</sub> because every coordinate system is expressed in AU, not in pixels. The Flower-of-Life circles (Tab 1, 3, 8, 11, 12), the 30-unit dimensional grid (Tab 9, 10), the scope-gate positions (Python <code>gxy</code>), the tangent spheres at trajectory gates, the 3D FoL sphere lattice &mdash; all of them rescale uniformly. Reference orbits at 0.72 / 1.0 / 1.52 AU stay at their physical radii (heliocentric convention), so when R<sub>s</sub> is large they appear as a small dot near FTOP, which is exactly the correct depiction of a far-out scene.</div>

<h3 style="color:#ffaa44">FTOP vs. CENTRAL MASS (important correction)</h3>
<div class="er" style="color:#ffaa44;border:1px solid #ffaa44;padding:6px;border-radius:4px">
<b>FTOP is NOT a gravitating body.</b> It is the <i>Focal Tensor Orientation Point</i> &mdash; an abstract (0,0) used as:<br>
&nbsp;&nbsp;&bull; the origin of the Flower-of-Life template (where the 31 circles are centred),<br>
&nbsp;&nbsp;&bull; the anchor of the 7 o'clock barrel orientation (the canonical position that puts the focused barrel at 240&deg;),<br>
&nbsp;&nbsp;&bull; the origin of the dimensional-sphere grid (&plusmn;30 units = &plusmn;R<sub>s</sub>).<br>
The <b>Sun</b>, by contrast, is a separate gravitating body (&mu;<sub>&odot;</sub> = 4&pi;&sup2; AU&sup3;/yr&sup2;) that sits at <code>sun_pos</code> in world AU. By default <code>sun_pos = (0,0)</code> so the Sun coincides with FTOP and the two-body solver gives the same answer it always did. But the code now separates the two concepts cleanly: you can place the Sun anywhere, and FTOP stays as the orientation anchor.
</div>

<h3 style="color:#88ff88">STARTUP &amp; PERFORMANCE</h3>
<div class="er" style="color:#ddd">Hit.py now boots in about 1&nbsp;s on a default run instead of ~60 s. The script prints a timed progress line on stdout for each init phase (construct, Monte Carlo, parallel+swarm, encode, server-ready), and the browser shows a gold-ringed <b>loading splash</b> while the JS parses the viz payload, builds the 31-circle lattice, resolves Fibonacci indices, wires handlers, and renders the first frame.</div>
<div class="er" style="color:#ddd"><b>What made it fast:</b></div>
<div class="er">1. <b>2-body fast path</b> &mdash; the RK4 integrator now uses a <i>scalar</i> step (<code>_rk4_scalar</code>) that carries state as 4 Python floats and calls <code>_g_xy</code> returning a tuple, eliminating millions of <code>np.array([...])</code> allocations in the hot loop. About 10&times; faster when only the Sun is gravitating.</div>
<div class="er">2. <b>Perturbations opt-in</b> &mdash; the n-body solar system (1 Sun + 20 planets + moons) costs ~25&times; more per RK4 substep, so <code>use_perturbations</code> now defaults to <b>False</b>. Toggle it on via <span class="ek">CONFIG &rarr; TARGET &rarr; N-body perturbations</span> when you want the full perturbed solution.</div>
<div class="er">3. <b>Lower default Monte-Carlo N</b> &mdash; <code>--sims</code> default dropped from 2000 to 300 so the first view appears quickly; crank it up for publication runs.</div>
<div class="er">4. <b>Body-snapshot cache</b> &mdash; <code>_install_bodies()</code> only rebuilds the global <code>BODIES</code> list when the scene key (sun_pos, tp, solar) changes; repeated calls from <code>_solve</code> / <code>_gates</code> / <code>_parallel</code> / <code>_swarm</code> during one run hit the cache.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">python Hit.py                 # fast: 2-body, 300 sims,  ~1 s</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">python Hit.py --sims 2000     # thorough: 2-body,  ~5 s</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">python Hit.py --perturbations # n-body, 300 sims,  ~60 s  (opt in)</div>

<h3 style="color:#ff88cc">SOLAR SYSTEM MODEL + N-BODY PERTURBATIONS</h3>
<div class="er" style="color:#ddd">The physics engine now carries a full solar system: <b>1 Sun + 20 planets + 0&ndash;5 random moons per planet</b>. Every body is a gravitating source summed into the RK4 integrator&rsquo;s acceleration:</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">a(r) = &Sigma;<sub>bodies</sub>  &minus; &mu;<sub>i</sub> (r &minus; r<sub>i</sub>) / |r &minus; r<sub>i</sub>|&sup3;</div>
<div class="er" style="color:#ddd">Planet orbits are coplanar, circular, and log-spaced in radius from 0.39 AU (inner) to 40 AU (outer). Each planet&rsquo;s &mu; is log-uniform in [10<sup>&minus;9</sup>&mu;<sub>&odot;</sub>, 10<sup>&minus;3</sup>&mu;<sub>&odot;</sub>] &mdash; covering asteroid-class up to Jupiter-class bodies. Moons orbit their parent planet at 0.002&ndash;0.03 AU with &mu; scaled to 10<sup>&minus;4</sup> to 10<sup>&minus;2</sup> of the parent. The configuration is reproducible (seeded with 42) so every run uses the same 21-body system.</div>
<div class="er" style="color:#ddd"><b>Frozen snapshot:</b> the solver uses a snapshot of every body&rsquo;s position at t&nbsp;=&nbsp;0 held constant through the 0.7-year flight. Error vs. a full n-body propagation of the same scene is &lt; (planet_period / T<sub>f</sub>)<sup>&minus;1</sup>, i.e. negligible for outer planets. Inner-planet close passes are absorbed into the Newton shooter&rsquo;s v<sub>0</sub>. Set <code>use_perturbations=False</code> to collapse back to the 2-body Keplerian behaviour.</div>
<div class="er" style="color:#ddd"><b>Rendering:</b> every 2D/3D tab (1, 3, 7, 8, 11, 12) now draws the sun + 20 planets + moons as filled coloured dots with dashed orbital rings. Planet labels appear next to each body. On Tab 1 SCOPE a <b>left-click</b> on any planet or moon sets it as the TARGET &mdash; the POST to <code>/resim</code> includes <code>planet_idx</code> + <code>moon_idx</code> and the backend resolves the selector to an exact (x,y) before re-solving the transfer orbit.</div>
<div class="er" style="color:#ddd"><b>Hover on any body</b> shows its name, kind (sun/planet/moon), world-AU position, sphere-nav address, orbital radius + period, &mu; value, and &mu; as a fraction of the Sun&rsquo;s. Clicking promotes it to TARGET.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">/resim payload: {..., sun_x, sun_y, use_perturbations, planet_idx, moon_idx} &mdash; any subset can be omitted to keep the current value.</div>

<h3 style="color:#66ddff">SPHERE NAV &mdash; TRI-AXIS 360° ADDRESSING</h3>
<div class="er" style="color:#ddd">The navigation sphere is a <b>3D volume</b>, and a single 360&deg; azimuth can only describe points in one plane. The address now emits <b>three independent 360&deg; rotations</b> &mdash; one about each coordinate axis &mdash; so every point in the sphere volume has a uniquely recoverable orientation readout.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">addr = &quot;Z&phi;=X&deg;  Y&phi;=Y&deg;  X&phi;=Z&deg;, &plusmn;Xz, &plusmn;Yx, &plusmn;Zy&quot;</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Z-axis &phi;<sub>Z</sub> = atan2(gy, gx)  &isin; [0&deg;, 360&deg;)   &nbsp;rotation in the XY plane</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Y-axis &phi;<sub>Y</sub> = atan2(gz, gx)  &isin; [0&deg;, 360&deg;)   &nbsp;rotation in the XZ plane</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">X-axis &phi;<sub>X</sub> = atan2(gz, gy)  &isin; [0&deg;, 360&deg;)   &nbsp;rotation in the YZ plane</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">gx, gy, gz = (x, y, z) / Rs &times; 30    &nbsp;grid units, each axis &plusmn;30</div>
<div class="er" style="color:#ddd">Each rotation is the <b>projection of the position onto one coordinate plane</b> expressed as a bearing. For a point on the scope plane (z=0), &phi;<sub>Y</sub> and &phi;<sub>X</sub> collapse to 0&deg; or 180&deg; depending on the x / y sign; for an off-plane point they carry non-trivial values and together fully describe the direction vector. The three angles are redundant for pure radial-direction recovery (two would suffice) but the redundancy is deliberate: each axis has its own dashboard readout, so the user can see at a glance &quot;how rotated am I about each axis?&quot;</div>
<div class="er" style="color:#ddd"><b>Example:</b> for (x, y, z) = (&minus;1.03, &minus;0.40, 0) on the scope plane &mdash; Z&phi; = 201&deg; (most of the info, XY bearing), Y&phi; = 180&deg; (straight back along &minus;X in the XZ plane), X&phi; = 180&deg; (straight back along &minus;Y in the YZ plane). For (1, 1, 1) in 3D &mdash; Z&phi; = 45&deg;, Y&phi; = 45&deg;, X&phi; = 45&deg; (the point sits on the unit diagonal, 45&deg; in every plane).</div>
<div class="er" style="color:#ddd">Elevation &theta; = atan2(gz, &radic;(gx&sup2;+gy&sup2;)) &isin; [&minus;90&deg;, +90&deg;] is still computed and available on the sphere object (as <code>.elevation</code>), but the printed address uses the tri-axis format everywhere.</div>

<h3 style="color:#66ddff">TARGET Z &mdash; OFF-PLANE TARGETS ARE LEGAL</h3>
<div class="er" style="color:#ddd">Targets (and the Sun) now have an explicit <code>z</code> component, exposed through CONFIG&rsquo;s target section and the <code>/resim</code> payload&rsquo;s <code>tz</code> field. The physics engine continues to run in the XY scope plane (the flower-of-life stencil is a 2D slice of a 3D volume), so the transfer orbit is solved against the XY projection; but the <b>3D views (Tabs 3, 10, 12) render the target at (tx, ty, tz)</b> with a dashed vertical drop-line to its ground-plane shadow. A non-zero <code>tz</code> is shown in the target label, and the gate coordinates follow the projectile path which is in-plane. This matches the program&rsquo;s design principle: the 2D slice is authoritative, but 3D views support Z.</div>

<h3 style="color:#ffaa44">UNIFIED FIBONACCI CROSS-SECTION LABELING</h3>
<div class="er" style="color:#ddd">The cross-section intersections are <i>locations of the Fibonacci spiral</i> through the Flower-of-Life lattice. Tab 1 (SCOPE) computes them once at script load via exact analytical circle-circle intersections of the 31-circle lattice, sorts them by radial distance from FTOP then clockwise from 12 o&rsquo;clock, and highlights indices <b>{0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89}</b> (Fibonacci). Tab 3 (3D VIEW) previously recomputed its own lattice with a slightly different sphere set, producing a different XS list and therefore different <code>F#</code> labels (as shown in your image 3 vs image 4).</div>
<div class="er" style="color:#ddd"><b>Fixed:</b> Tab 3 now reads directly from the shared <code>XS</code> array + <code>FIB_SET</code> that Tab 1 uses, so every <code>F#</code> label is identical between the 2D slice and the 3D corridor. Tab 3 also draws the <b>Fibonacci spiral arc</b> through those points (dashed orange quadratic Bezier) mirroring what Tab 1 shows &mdash; so the 3D view now depicts the spiral as a 3D-projected curve on the z=0 slice (the only plane where the lattice exists).</div>

<h3 style="color:#66ddff">HOVER &mdash; SPHERE COORDS EVERYWHERE</h3>
<div class="er" style="color:#ddd">Every hover tooltip across every tab now includes the <b>dual-angle sphere-nav coordinate</b> (<code>&phi;&deg;, &theta;&deg;, gx, gy, gz</code>) for whatever body, gate, or trajectory sample is under the cursor, plus the 24-cycle time stamp. The shared helper <code>sphereAddr(x, y, z=0)</code> computes it for any 3D point by mapping <code>R<sub>s</sub> &rarr; 30 grid units</code>. When a hover is inside the scope the <code>(in)</code> flag is shown; outside it prints <code>(out)</code>.</div>
<div class="er" style="color:#ddd">Gate tooltips also include: velocity components (vx, vy), heading AND &Delta;heading, &Delta;speed, J-condition, arc length, R(FTOP), d(Barrel), d(Target), time-to-go, and gravity magnitude. Any single hover answers every question about that point in one glance.</div>

<h3 style="color:#88ddff">3D FILLED FLOWER-OF-LIFE SPHERE GRID</h3>
<div class="er" style="color:#ddd">The 2D scope shown in Tab 1 is a <b>cross-section slice</b> of a denser 3D expression. In every 3D tab (3 &middot; 7 &middot; 8 &middot; 10) the space around the corridor is filled with transparent spheres arranged as a Flower-of-Life lattice extruded into Z &mdash; a volumetric grid you see through, identical in topology to the 2D slice but now filling a ball of space. Tab 9 (Sphere Nav) keeps the 2D nav-ball depiction: the same lattice, flattened to the scope plane for fast grid addressing.</div>
<div class="er" style="color:#ddd">Spheres are drawn as three orthogonal wireframe rings (XY &middot; XZ &middot; YZ) at very low alpha so the trajectory, gates, barrel and target remain fully legible. The lattice tiles outward from FTOP toward the target and vertically on z = &plusmn;r<sub>1</sub>&radic;3/2 layers with a half-unit hex offset &mdash; close-packed, the same way real flower-of-life extends in 3D.</div>
<h3>ARCHITECTURE</h3>
<div class="er">1. Python backend: physics engine, Monte Carlo, parallel scope, swarm solver</div>
<div class="er">2. Embedded HTTP server: serves single-page HTML dashboard</div>
<div class="er">3. <b>12 tabs:</b> SCOPE (2D tensor), IMPACT (scatter), 3D VIEW (corridor), STATS (extreme total system overview), CONSOLE (log), CONFIG (parameters), SALVO (multi-barrel), TRAJECTORY (Soulscape renderer), SPHERE NAV (dimensional coordinate system), 3D SPHERE (3D nav ball), PARALLEL (parallel scope at main-scope fidelity), PARALLEL 3D (parallel corridor in 3D)</div>
<div class="er">4. All visualization data computed once, injected as JSON into HTML</div>
<h3>CONTROLS</h3>
<div class="er"><span class="ek">1-9</span>Switch tabs 1..9 (SCOPE, IMPACT, 3D, STATS, CONSOLE, CONFIG, SALVO, TRAJECTORY, SPHERE NAV). <span class="ek">0</span>=10 3D SPHERE. Tabs 11 PARALLEL and 12 PARALLEL 3D are click-only.</div>
<div class="er"><span class="ek">ESC</span>Toggle this information panel</div>
<div class="er"><span class="ek">SPACE</span>Play / Pause live trajectory animation</div>
<div class="er"><span class="ek">&larr; &rarr;</span>Step backward/forward one gate interval</div>
<div class="er"><span class="ek">T</span>Toggle target input panel (set X,Y in AU)</div>
<div class="er"><span class="ek">P</span>Toggle parallel offset scope overlay</div>
<div class="er"><span class="ek">S</span>Toggle multi-barrel swarm overlay</div>
<div class="er"><span class="ek">+/-</span>Zoom scope view in/out</div>
<div class="er"><span class="ek">Scroll</span>Zoom toward mouse cursor (scope) or zoom 3D view</div>
<div class="er"><span class="ek">RClick+Drag</span>Pan scope in world coordinates &mdash; all elements move together</div>
<div class="er"><span class="ek">Drag</span>Rotate 3D view</div>
<div class="er"><span class="ek">R</span>Reset camera to origin (scope + 3D)</div>
<div class="er"><span class="ek">D</span>Toggle Display Settings panel (font size + layer visibility)</div>
<div class="er"><span class="ek">Hover</span>Tooltip data on any gate, cross-section, or landmark</div>
<h3 style="color:#4488ff">CAMERA MODEL (Simulation.py)</h3>
<div class="er" style="color:#ddd">The scope uses a <b>world-coordinate camera</b> (camX, camY in AU, camZ = zoom multiplier).</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">screenPos = (worldPos - cam) &times; baseSc &times; camZ + center</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">worldPos = (screenPos - center) / (baseSc &times; camZ) + cam</div>
<div class="er" style="color:#ddd"><b>Zoom toward mouse:</b> When scrolling, the world point under the cursor is preserved. The camera position adjusts so that point stays fixed on screen while everything else scales around it.</div>
<div class="er" style="color:#ddd"><b>Scale range:</b> 0.005x (intergalactic) to 200x (sub-AU detail). At very low zoom, the Fibonacci labels LOD out; at galactic scale, the ruler switches to parsecs.</div>
<div class="er" style="color:#ddd"><b>Pan:</b> Right-click drag moves the camera in world coordinates: cam -= dx/scale. This means panning speed is proportional to zoom level. ALL elements (target, barrel, scope, Fibonacci, gates) move together &mdash; they are all in AU world space.</div>
<h3 style="color:#c8a96e">INFINITY MAPPING</h3>
<div class="er" style="color:#ddd">Conceptually, the scope maps infinity to a finite navigable canvas. At zoom 1x, ~2.8 AU is visible. Zooming out to 0.005x shows ~560 AU &asymp; 0.003 pc. The Flower-of-Life lattice, reference orbits, and all trajectory data remain mathematically accurate at any scale.</div>
<div class="er" style="color:#ddd">The dimensional mapping follows: <b>360&deg; sphere &times; 60&times;60 grid &times; 60 depth layers &times; 24 cycles = 5,184,000 sphered locations</b>. At any zoom level, the scope renders the subset of this space visible in the current viewport.</div>
<div class="er" style="color:#ddd">Scale labels adapt automatically: sub-AU shows km, normal shows AU, galactic shows kAU or parsecs.</div>
<h3>PHYSICS ENGINE</h3>
<div class="er" style="color:#ddd">Gravity: <b>a = -&mu;r/|r|&sup3;</b> where &mu; = 4&pi;&sup2; &asymp; 39.478 AU&sup3;/yr&sup2;</div>
<div class="er">Integrator: 4th-order Runge-Kutta, dt = 0.005 yr (~1.83 days)</div>
<div class="er">Transfer solution: Newton shooting with 17&times;17 grid search initialization</div>
<div class="er">State vector: [x, y, v<sub>x</sub>, v<sub>y</sub>] &mdash; 4D phase space</div>
<div class="er">State Transition Matrix: &Phi;(t) accumulated across 13 segments</div>
<div class="er">Tensor [T] = &frac12;(&Phi;+&Phi;<sup>T</sup>), normalized by &radic;|diag| &mdash; symmetric correlation</div>
<div class="er">Gate corrections: &Delta;v = -J<sup>-1</sup><sub>v</sub> &middot; (predicted miss) &middot; damping</div>
<div class="er">Damping: 0.7 &times; max(0.2, (12-k)/12) per gate k &mdash; decays late</div>
<div class="er">Max &Delta;v: 0.05 AU/yr per gate &mdash; thrust-limited</div>
<div class="er">Navigation noise: &sigma;<sub>pos</sub>=0.01 AU, &sigma;<sub>vel</sub>=0.001 AU/yr</div>
<h3>TARGET &amp; FTOP</h3>
<div class="er" style="color:#44ff88">Target: (1.520, 0.000) AU &mdash; outer reference orbit zone</div>
<div class="er">Hit radius: 0.02 AU (~2,992,000 km)</div>
<div class="er">Purpose: energy delivery via redirected comet/meteor impact</div>
<div class="er" style="color:#ffcc00">&#9674; FTOP &mdash; Focal Tensor Orientation Point</div>
<div class="er" style="color:#aaa">The FTOP is the gravitational data mass at coordinate origin. It represents any orientation center: solar, galactic, or arbitrary. All measurements are relative to FTOP. Right-click panning moves the viewport around FTOP while preserving AU coordinate accuracy.</div>
<h3>FLOWER-OF-LIFE GEOMETRY</h3>
<div class="er" style="color:#ddd">The scope uses a Flower-of-Life lattice as its structural framework:</div>
<div class="er">1 center circle (r = Rs&times;0.58) + 6 inner petals (same radius, offset by Rs&times;0.58)</div>
<div class="er">12 scope-ring circles (r = Rs, centered at Rs from FTOP, 30&deg; apart)</div>
<div class="er">12 outer circles (r = Rs, centered at Rs&times;1.5, offset 15&deg;)</div>
<div class="er">Total: 31 circles forming the complete Flower-of-Life lattice</div>
<h3 style="color:#ffaa44">CROSS-SECTION FIBONACCI PROOF</h3>
<div class="er" style="color:#ddd">Every pair of the 31 circles is tested for intersection using the analytical circle-circle formula:</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">d = |c1-c2|, a = (r1&sup2;-r2&sup2;+d&sup2;)/(2d), h = &radic;(r1&sup2;-a&sup2;)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">P = midpoint &plusmn; h&times;perpendicular &rarr; 0 or 2 intersection points</div>
<div class="er" style="color:#ddd">C(31,2) = 465 pairwise tests. Each valid intersection produces exactly 2 vesica piscis loci. Points within &epsilon;=0.02 AU are merged (shared vertices).</div>
<div class="er" style="color:#ddd"><b>Enumeration:</b> All unique points are sorted by (1) radial distance from FTOP, then (2) clockwise angle from 12 o&rsquo;clock. This creates a natural outward spiral.</div>
<div class="er" style="color:#ffaa44"><b>Fibonacci overlay:</b> Points at spiral indices {0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89} receive bold orange rings. These are the Fibonacci numbers F(n) = F(n-1) + F(n-2).</div>
<div class="er" style="color:#ddd"><b>Proof of completeness:</b> Since every circle pair is tested analytically, NO intersection point is missed. Every vesica piscis locus of the lattice is accounted for. There are no &ldquo;empty variables&rdquo; &mdash; every cross-section position is computed, deduplicated, indexed, and displayed with its spiral number.</div>
<div class="er" style="color:#ddd"><b>Pattern significance:</b> The Fibonacci spiral through the Flower-of-Life cross-sections mirrors the golden ratio &phi; = (1+&radic;5)/2 &asymp; 1.618 inherent in sacred geometry. The bold rings mark structurally significant nodes where the spiral&rsquo;s growth rate matches the lattice spacing.</div>
<h3>REFERENCE ORBITS</h3>
<div class="er" style="color:#ccaa66">&#9675; Inner reference &mdash; 0.72 AU (Venus-analog zone)</div>
<div class="er" style="color:#6699ff">&#9675; Mid reference &mdash; 1.00 AU (Earth-analog zone)</div>
<div class="er" style="color:#ff8866">&#9675; Outer reference &mdash; 1.52 AU (Mars-analog / target zone)</div>
<h3 style="color:#00dd88">PARALLEL OFFSET SCOPE</h3>
<div class="er" style="color:#ddd">The parallel scope is a second trajectory computed from a barrel offset by a configurable angle (default +5&deg;). Both trajectories target the same point.</div>
<div class="er" style="color:#ddd"><b>Δ vectors:</b> At each gate, the difference (parallel - primary) in position and velocity is computed. These Δ values represent the trajectory's <b>sensitivity</b> to barrel placement — they are the directional derivatives of the solution with respect to angular offset.</div>
<div class="er" style="color:#ddd"><b>Corrective use:</b> During the pre-comet phase, if actual barrel placement deviates from nominal, the parallel Δ data provides pre-computed correction magnitudes. Formula pattern: <b>(A-X+(A*X))*(+A-X)*(1/2)*(0/1)</b> — the product of sum and difference with halving selects the corrective mode between parallel and primary.</div>
<div class="er"><span class="ek">P</span>Toggle parallel scope overlay on/off</div>
<div class="er" style="color:#ddd"><b>Dedicated PARALLEL tab (11):</b> the parallel scope rendered as the primary of its own scope, with full tensor lattice, reference orbits, 12 gate markers labeled with live Δpos / Δvel / Δhdg, an animated parallel comet, and a right-side live-math panel carrying the same orbital-mechanics dossier (ε, L, a, e, γ, v/v<sub>esc</sub>, jerk) plus Δε / ΔL / Δa / Δe vs primary. Drag = pan, wheel = zoom-toward-cursor. Primary trajectory stays visible dimly as a reference.</div>
<div class="er" style="color:#ddd"><b>PARALLEL 3D tab (12):</b> tab 3 applied to the parallel barrel — full 3D corridor, FoL lattice (green-tinted), speed-colored parallel trajectory, gate tangent spheres with Δ labels, animated parallel comet on the Z=time axis, and the primary corridor kept dim as a reference. Independent rotation/zoom state (does not disturb tab 3). Left-drag rotates, scroll zooms, R resets all rotations.</div>
<h3 style="color:#cc44ff">MULTI-BARREL SWARM</h3>
<div class="er" style="color:#ddd">The swarm system places barrels at multiple clock positions (1h, 3h, 5h, 9h, 11h) on the scope ring. Each barrel independently solves a transfer orbit to the <b>same target X</b>.</div>
<div class="er" style="color:#ddd"><b>Convergence:</b> All swarm trajectories converge on the target. When the target moves, ALL barrels auto-recalculate. This is the "reverse" of the single-barrel concept — instead of one comet path, it's a group of meteors all aimed at one location.</div>
<div class="er" style="color:#ddd"><b>Combined probability:</b> P(hit) = 1 - &prod;(1 - p<sub>i</sub>). With 5 independent barrels each at ~99% corrected hit rate, the combined probability approaches 100%.</div>
<div class="er" style="color:#ddd"><b>Design decision:</b> The green lines in the reference imagery show convergence paths from perimeter nodes to a central target. This is implemented as independent Newton-shooting solutions from each clock position, each with its own v&sub;0 and gate corrections.</div>
<div class="er"><span class="ek">S</span>Toggle swarm overlay on/off</div>
<h3>SCOPE PANNING &amp; ZOOM</h3>
<div class="er" style="color:#ddd">Right-click drag pans the viewport in world coordinates. Scroll zooms toward the mouse cursor. The coordinate transform: <b>screen = center + (world - cam) &times; baseSc &times; camZ</b></div>
<div class="er" style="color:#ddd">All objects (target, barrel, scope ring, gates, Fibonacci lattice, reference orbits) exist in a single AU coordinate space. They ALL pan and zoom together as one unified view. There is no separation between "fixed" and "moving" objects.</div>
<div class="er" style="color:#ddd"><b>Zoom range:</b> 0.005x (see ~560 AU, intergalactic) to 200x (sub-AU detail). Scale ruler auto-adapts: km &rarr; AU &rarr; kAU &rarr; parsec.</div>
<div class="er" style="color:#ddd"><b>LOD:</b> At low zoom, non-Fibonacci labels hide. At very low zoom, Fibonacci labels hide too. Gate labels, orbit labels, and barrel details remain visible at their appropriate scales.</div>
<h3 style="color:#ff8866">TAB 6: CONFIG</h3>
<div class="er" style="color:#ddd">Full parameter customization: barrel position/angle, target position, comet/meteor properties, flight parameters, physics constants, gate configuration, noise model, display options.</div>
<h3 style="color:#c8a96e">TAB 7: SALVO</h3>
<div class="er" style="color:#ddd">Combined multi-barrel convergence graph. Shows ALL barrels (primary + parallel + 5 swarm = 7 total) on one unified scope view with trajectories overlaid, convergence lines, and a right-side info panel with per-barrel stats and combined hit probability.</div>
<div class="er" style="color:#ddd"><b>Barrel Focus:</b> Click any barrel (on canvas or in info panel) to scope to it. The focused barrel is always rendered at <b>7 o&rsquo;clock</b> (240&deg;), and all other barrels shift in formation relative to it.</div>
<div class="er" style="color:#ddd"><b>Fibonacci gate numbering:</b> Gates at Fibonacci positions (1, 2, 3, 5, 8) are highlighted with glow rings and labeled F1&ndash;F8 on each trajectory.</div>
<div class="er" style="color:#ffd700;border:1px solid #ffd700;padding:6px;border-radius:4px;margin-top:6px"><b>NOTICE*</b> &mdash; When any individual salvo barrel is focused/scoped, that barrel is ALWAYS positioned at 7 o&rsquo;clock (240&deg;). All other barrels maintain their relative formation &mdash; they rotate as a group so the focused barrel sits at the primary position. Click again to unfocus and return to absolute positioning.</div>
<h3 style="color:#ff8800">TAB 8: TRAJECTORY (Soulscape)</h3>
<div class="er" style="color:#ddd">Atmospheric canvas renderer using Soulscape graphics: dark gradient background, twinkling stars, gravity field visualization, triple-pass glow trajectory, golden gate markers, pulsing target, ember particle system on comet, WoW-style HUD frame with progress bar, and live data overlay.</div>
<div class="er" style="color:#ffd700;border:1px solid #ffd700;padding:6px;border-radius:4px;margin-top:6px"><b>ACTIVE BARREL SELECTION</b> &mdash; Tab 8 is now a <i>per-barrel</i> trajectory inspector. Every barrel in the array (PRIMARY, PARALLEL, SWARM@1h/3h/5h/9h/11h) can be promoted to the <b>PRIMARY view of this tab</b>. When promoted, it:
<ul style="margin:4px 0 0 16px;padding:0;color:#ddd;font-size:11px">
<li>sits at the <b>barrel position</b> with a bright swatch colored to match it,</li>
<li>shows <b>its own</b> trajectory (glow + speed-colored core + gate markers + velocity arrows),</li>
<li>is labeled <b>&quot;PRIMARY (<i>origLbl</i>)&quot;</b> in the HUD, info bar, and tooltip,</li>
<li>drives the ember-particle animated comet along its own path,</li>
<li>propagates to <b>all other tabs</b> (SALVO focus rotation at 7h, STATS table, CONFIG projectile list, SCOPE HUD, PARALLEL tab), so the choice is global.</li>
</ul></div>
<div class="er" style="color:#ddd"><b>Non-primary barrels</b> remain visible on Tab 8 as dim dashed trajectories + grey-ringed markers labeled &quot;NON-PRIMARY&quot; so you can see how the other firing solutions compare. Click any of them (or the matching chip in the top-right picker row) to make it the new PRIMARY.</div>
<div class="er" style="color:#ddd"><b>How to pick:</b>
<ul style="margin:4px 0 0 16px;padding:0;color:#ddd;font-size:11px">
<li>Click a chip in the top-right picker row of Tab 8.</li>
<li>Click any barrel marker on the Tab 8 canvas.</li>
<li>Click any barrel marker or info-panel row in Tab 7 SALVO.</li>
</ul>
All three surfaces share the same <code>salvoFocus</code> state; flipping it in one flips it everywhere. Click the currently-active chip to keep it (it stays active); promote any other to switch. If the salvo is unfocused (<code>salvoFocus = &minus;1</code>), the original PRIMARY barrel is the active one by default.</div>
<div class="er" style="color:#ddd"><b>Role labels:</b> the globally-propagated <code>lbl</code> field on each barrel now reads <code>&quot;PRIMARY (<i>origLbl</i>)&quot;</code> for the active one and <code>&quot;NON-PRIMARY (<i>origLbl</i>)&quot;</code> for all others &mdash; visible in the SALVO side panel, the STATS barrel-array table, the salvo &quot;FIRING&quot; banner, and tooltips. The original identity (PRIMARY / PARALLEL / SWARM@Nh) is preserved as <code>origLbl</code> so no information is lost.</div>
<h3 style="color:#44ffaa">TAB 9: SPHERE NAV (Dimensional Sphere Coordinates)</h3>
<div class="er" style="color:#ddd">Alternative coordinate visualization mapping the entire scope to a <b>dimensional sphere grid</b> with address format: <b>X&deg;,xz,xx,xy</b></div>
<div class="er" style="color:#ddd"><b>Infinity = Scope:</b> The scope radius Rs is treated as &infin;. All positions within Rs map to grid coordinates &plusmn;30. Scale: 1 grid unit = Rs/30 AU.</div>
<div class="er" style="color:#ddd"><b>Grid:</b> 60 divisions per axis (&minus;30 to +30) &times; 60 &times; 60 &times; 24 time cycles = <b>5,184,000 addressable sphere locations</b>.</div>
<div class="er" style="color:#ddd"><b>Time cycles:</b> Flight time (0 to T<sub>f</sub>) maps to 24 looped cycles (00:00:00 &ndash; 23:59:59). Each cycle fraction corresponds to mission progress.</div>
<div class="er" style="color:#ddd"><b>Rendering:</b> All objects (trajectory, gates, barrel, target, swarm, parallel) are mapped into sphere grid coordinates. Hover shows live sphere address. Angular ticks 0&deg;&ndash;360&deg; around scope boundary. Quadrant labels match the xz,xx,xy notation.</div>
<div class="er" style="color:#ddd"><b>Use case:</b> Universal coordinate system for any located planet, mass, or celestial body within the scope. Scrollable/zoomable to show all data points. When zoomed out, the entire observable scope is mapped; when zoomed in, sub-grid resolution provides precise locating.</div>
<h3 style="color:#ffd700;border:1px solid #ffd700;padding:4px 8px;border-radius:4px;margin-top:8px">&#9733; COMET &#8594; METEOR REDIRECTION &mdash; COMPLETE SYSTEM GUIDE</h3>
<div class="er" style="color:#c8a96e;font-size:11px"><b>Mission:</b> A freely-placed barrel (pusher) adjusts a passing comet into a precise meteor trajectory aimed at a target. FTOP is always the orientation center. The barrel&rsquo;s default focus position is 7 o&rsquo;clock (240&deg;), but it can be placed at ANY coordinate in AU space.</div>
<h4 style="color:#4488ff;margin:8px 0 4px">ORIENTATION MODEL</h4>
<div class="er" style="color:#ddd"><b>FTOP (Focal Tensor Orientation Point)</b> sits at coordinate origin (0, 0). It is the gravitational/computational anchor &mdash; think of it as the Sun or any central mass. All distances, angles, and trajectories are measured relative to FTOP.</div>
<div class="er" style="color:#ddd"><b>7 o&rsquo;clock = 240&deg;</b> is the <i>canonical focus orientation</i>. When you focus a barrel in the SALVO view, it always renders at 7h and all other barrels orbit relative to it &mdash; maintaining true relative geometry. This is a display convention only; the physics uses raw AU coordinates.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Clock angle: &theta; = 90&deg; &minus; (clock &times; 30&deg;) &nbsp;|&nbsp; 7h &rarr; &theta; = 90&deg; &minus; 210&deg; = &minus;120&deg; &equiv; 240&deg;</div>
<div class="er" style="color:#ddd"><b>Barrel placement is completely free.</b> Enter any (X, Y) in AU, or snap to a clock position on the scope ring (r = Rs). The Newton shooting solver recomputes the full transfer orbit from that exact position regardless of where it is &mdash; on the ring, inside it, or far outside it.</div>
<h4 style="color:#44ff88;margin:8px 0 4px">COMET &#8594; METEOR CONVERSION</h4>
<div class="er" style="color:#ddd">A <b>comet</b> is a natural body on an existing Keplerian orbit (elliptical, parabolic, or hyperbolic). A <b>meteor</b> is a comet that has been given a precision &Delta;v nudge so it impacts a specific target. The barrel is the delivery mechanism for that &Delta;v.</div>
<div class="er" style="color:#ddd"><b>Step 1 &mdash; Detect:</b> The comet is tracked. Its initial state (position + velocity) is measured. The target&rsquo;s position and orbital motion are computed forward to the intercept time T<sub>f</sub>.</div>
<div class="er" style="color:#ddd"><b>Step 2 &mdash; Solve:</b> Newton shooting finds the initial velocity v&sub;0 the comet needs at the barrel position to arrive at the target at T<sub>f</sub>. The &Delta;v required = v&sub;0 &minus; v<sub>comet_current</sub>. This is the energy budget.</div>
<div class="er" style="color:#ddd"><b>Step 3 &mdash; Aim:</b> The 12 correction gates along the trajectory provide mid-course adjustments. Each gate applies a small &Delta;v (damped, clamped at 0.05 AU/yr) using precomputed Jacobian sensitivity maps. The comet is steered like a ship through waypoints.</div>
<div class="er" style="color:#ddd"><b>Step 4 &mdash; Confirm:</b> Monte Carlo simulations (N=500&ndash;2000 runs) with position/velocity noise quantify hit probability. Baseline (no corrections) vs corrected (12-gate) rates show the improvement factor.</div>
<h4 style="color:#ff8866;margin:8px 0 4px">AIMING &amp; BARREL PLACEMENT STRATEGY</h4>
<div class="er" style="color:#ddd">The barrel position determines the <b>transfer orbit geometry</b>. Different positions produce different approach angles, flight times, and energy costs:</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Optimal barrel: minimises |&Delta;v| = |v&sub;0 &minus; v<sub>comet</sub>| + &Sigma;|&Delta;v<sub>gate</sub>|</div>
<div class="er" style="color:#ddd">&bull; <b>Closer barrel:</b> shorter flight, higher required v&sub;0, steeper approach angle &rarr; harder to correct</div>
<div class="er" style="color:#ddd">&bull; <b>Farther barrel:</b> longer flight, lower v&sub;0 (Hohmann-like), shallower approach &rarr; more gate correction time</div>
<div class="er" style="color:#ddd">&bull; <b>Multiple barrels (swarm):</b> independent trajectories from N positions &rarr; combined P(hit) = 1 &minus; &prod;(1&minus;p<sub>i</sub>). Even if individual accuracy is 90%, 5 independent barrels give &gt;99.999% combined probability.</div>
<div class="er" style="color:#ddd">&bull; <b>Parallel barrel:</b> offset by a small angle from primary. The &Delta; at each gate = trajectory sensitivity to barrel placement. If actual barrel placement deviates, these &Delta; values give pre-computed corrections.</div>
<h4 style="color:#cc44ff;margin:8px 0 4px">ENERGY &amp; TRAJECTORY MATHEMATICS</h4>
<div class="er" style="color:#ddd"><b>Transfer orbit energy:</b> E = &minus;&mu;/(2a) where a = semi-major axis of the transfer ellipse. The barrel supplies kinetic energy KE = &frac12;mv&sub;0&sup2; to achieve the required v&sub;0.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">KE (Joules) = &frac12; &times; m(kg) &times; (v&sub;0 &times; 1.496&times;10&sup{11} / 3.156&times;10&sup7;)&sup2;</div>
<div class="er" style="color:#ddd"><b>Tsiolkovsky rocket equation:</b> &Delta;v = v<sub>e</sub> ln(m&sub;0/m<sub>f</sub>). For a pusher with exhaust velocity v<sub>e</sub> = 30 km/s, achieving &Delta;v = 1 km/s requires m&sub;0/m<sub>f</sub> = e&sup{1/30} &asymp; 1.034 &mdash; only 3.4% fuel mass fraction.</div>
<div class="er" style="color:#ddd"><b>Relativistic correction</b> (&beta; = v/c &lt; 10&sup{&minus;4}): Lorentz factor &gamma; = 1/&radic;(1&minus;&beta;&sup2;) &asymp; 1 + &beta;&sup2;/2. Time dilation: proper time on board = T<sub>f</sub>/&gamma;. At these speeds, relativistic effects are ~10&sup{&minus;8} &mdash; negligible in navigation but tracked for completeness.</div>
<div class="er" style="color:#ddd"><b>Gate Jacobian correction formula:</b> At gate k, miss deviation &Delta;r = r<sub>actual</sub> &minus; r<sub>nominal</sub>. The velocity block J<sub>v</sub> = J[k][:,2:4] solves &Delta;v<sub>k</sub> = &minus;J<sub>v</sub>&sup{&minus;1} &middot; &Delta;r, damped by 0.7&times;(12&minus;k)/12 so corrections decay toward intercept.</div>
<div class="er" style="color:#ddd"><b>Gravity:</b> a = &minus;&mu;r/|r|&sup3;, &mu; = 4&pi;&sup2; AU&sup3;/yr&sup2;. At 1 AU: |a| = 4&pi;&sup2; &asymp; 39.5 AU/yr&sup2; &asymp; 5.93 mm/s&sup2;. At target orbit (1.52 AU): |a| = 17.1 AU/yr&sup2; &asymp; 2.56 mm/s&sup2;.</div>
<div class="er" style="color:#ddd;border:1px solid #cc44ff;padding:6px;border-radius:4px;margin-top:4px"><b>BARREL PLACEMENT GUIDE:</b> Use CONFIG tab &rarr; PRIMARY BARREL section to set X,Y freely. Live readout shows r, angle, and clock equivalent. Click &ldquo;SNAP TO CLOCK&rdquo; to lock to scope ring. Each SWARM barrel has its own X/Y inputs in the SWARM table &mdash; edit any barrel independently for optimal geometric coverage of the target.</div>
<h3 style="color:#88ddff">LIVE MATH REFERENCE (every HUD field, defined)</h3>
<div class="er" style="color:#ddd">Each item below appears live in the SCOPE HUD (tab 1), 3D SCOPE STATUS (tab 3), and where applicable in the parallel and parallel-3D HUDs. The formulas are exact; no fits or empirical constants are used.</div>
<div class="er" style="color:#c8a96e;margin-top:6px"><b>KINEMATICS / CALCULUS</b></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Position r = (x, y)   |   Speed |v| = &radic;(vx&sup2;+vy&sup2;)   |   r&middot;v = x&middot;vx + y&middot;vy   (sign: +outbound / &minus;inbound)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Radial  v<sub>r</sub> = (r&middot;v)/|r|    Tangential  v<sub>t</sub> = (x&middot;vy &minus; y&middot;vx)/|r|</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Acceleration  a = &minus;&mu; r / |r|&sup3;,  |a| = &mu;/|r|&sup2;  (mm/s&sup2; conversion in the HUD)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Jerk  da/dt = &minus;&mu; (v/|r|&sup3; &minus; 3 r (r&middot;v) / |r|&sup5;)  &rarr; used for Newton-Raphson predictive control tests</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Curvature  &kappa;(s) = |&Delta;&theta;|/&Delta;s  (per-gate discrete sample)</div>

<div class="er" style="color:#c8a96e;margin-top:6px"><b>ORBIT MECHANICS (vis-viva + elements)</b></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Specific energy  &epsilon; = &frac12;|v|&sup2; &minus; &mu;/|r|</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Angular momentum  L = x&middot;vy &minus; y&middot;vx   (z-component, planar)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Semi-latus rectum  p = L&sup2;/&mu;</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Semi-major axis  a = &minus;&mu;/(2&epsilon;)    Period  T = 2&pi;&radic;(a&sup3;/&mu;)    Mean motion  n = &radic;(&mu;/a&sup3;)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Eccentricity vector  e = (v&times;L<sub>z</sub>)/&mu; &minus; r/|r|    |e| gives e  (0=circle, 1=parabola)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Arg of periapsis  &omega; = atan2(e<sub>y</sub>, e<sub>x</sub>)    True anomaly  &nu; = angle(r, e)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Periapsis  r<sub>p</sub> = a(1&minus;e)    Apoapsis  r<sub>a</sub> = a(1+e)  (closed orbits only)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Flight-path angle  &gamma; = atan2(v<sub>r</sub>, |v<sub>t</sub>|)   &rarr; 0&deg; for circular, ~45&deg; near apsis, signed by v<sub>r</sub></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Escape speed  v<sub>esc</sub> = &radic;(2&mu;/|r|)    Ratio v/v<sub>esc</sub> &lt; 1 bound, &ge; 1 unbound</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Vis-viva residual  &Delta;<sub>VV</sub> = v&sup2; &minus; &mu;(2/r &minus; 1/a)   &rarr; exact 2-body solutions give 0; non-zero exposes numerical drift</div>

<div class="er" style="color:#c8a96e;margin-top:6px"><b>TIDAL / GRAVITY-GRADIENT TENSOR (rank-2)</b></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">U<sub>xx</sub> = &minus;&mu;/|r|&sup3; + 3&mu;x&sup2;/|r|&sup5;    U<sub>yy</sub> = &minus;&mu;/|r|&sup3; + 3&mu;y&sup2;/|r|&sup5;    U<sub>xy</sub> = 3&mu;xy/|r|&sup5;</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">tr(&nabla;&nabla;U) = U<sub>xx</sub>+U<sub>yy</sub> = 0 in vacuum (Laplace); the HUD prints it as a numerical-zero check.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">det(&nabla;&nabla;U) = U<sub>xx</sub>U<sub>yy</sub> &minus; U<sub>xy</sub>&sup2;   &rarr; eigenvalue product; sign shows saddle vs. bowl topology of the gradient tensor.</div>

<div class="er" style="color:#c8a96e;margin-top:6px"><b>RELATIVISTIC / SPECIAL</b></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">&beta; = v/c, c = 63241.1 AU/yr   &rarr;   &gamma; = 1/&radic;(1&minus;&beta;&sup2;)   &rarr;   &tau;/T = 1/&gamma;  (proper-time dilation)</div>

<div class="er" style="color:#c8a96e;margin-top:6px"><b>GATE NUMERICS (per-gate k)</b></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">J-cond  &kappa;<sub>J</sub> = &sigma;<sub>max</sub>(J)/&sigma;<sub>min</sub>(J)   (2&times;2 velocity Jacobian condition number; high &rarr; correction ill-posed)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Damping factor  d<sub>k</sub> = 0.7 &times; max(0.2, (12&minus;k)/12)   &rarr; late-gate corrections are squeezed so we don&rsquo;t overshoot.</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Clamp  |&Delta;v<sub>k</sub>| &le; 0.05 AU/yr  (thrust-limited maneuver)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">Arc length  s<sub>k</sub> = |r<sub>k</sub>&minus;r<sub>k&minus;1</sub>|, &Delta;hdg<sub>k</sub>, &Delta;spd<sub>k</sub>  feed STATS &ldquo;CORRELATIONS&rdquo; card directly.</div>

<div class="er" style="color:#c8a96e;margin-top:6px"><b>PARALLEL &Delta; (+offset)</b></div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">&Delta;pos<sub>k</sub> = |r<sup>par</sup><sub>k</sub> &minus; r<sup>prim</sup><sub>k</sub>|    &Delta;vel<sub>k</sub> = |v<sup>par</sup><sub>k</sub> &minus; v<sup>prim</sup><sub>k</sub>|    &Delta;hdg<sub>k</sub> (deg)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">These are finite-difference approximations of &part;(solution)/&part;(barrel angle) &mdash; directional derivatives of the transfer orbit.</div>

<div class="er" style="color:#c8a96e;margin-top:6px"><b>STATS &mdash; CORRELATIONS MATRIX</b></div>
<div class="er" style="color:#ddd">The STATS tab now includes a 10&times;10 Pearson <code>r</code> matrix over every pairwise combination of the 12-gate series (<code>spd, rsun, hdg, dspd, dhdg, curv, arc, jc, grav, tg</code>). Heatmap colouring: gold positive, blue negative; |r| &ge; 0.8 cells are bolded. A narrative block lists every strong coupling below the table (e.g. &ldquo;spd &harr; rsun: r=&minus;0.998&rdquo; reveals the perfectly anti-correlated <i>slingshot signature</i>). A second table tracks <b>cross-barrel</b> correlations vs. the primary gate-speed profile.</div>
<div class="er" style="color:#ddd"><b>Gravity model:</b> 2-body Keplerian: <b>a = &minus;&mu;r/|r|&sup3;</b> where &mu; = 4&pi;&sup2; AU&sup3;/yr&sup2; (heliocentric). No n-body perturbations in current model.</div>
<div class="er" style="color:#ddd"><b>Integrator:</b> 4th-order Runge-Kutta (RK4) with step dt=0.005 yr. Error per step: O(dt&sup5;) &asymp; 3&times;10&sup{&minus;13} AU. Cumulative error over T<sub>f</sub>=0.7yr: &lt;10&sup{&minus;10} AU.</div>
<div class="er" style="color:#ddd"><b>Transfer orbit solver:</b> Newton shooting method. Grid search (17&times;17=289 initial guesses) &rarr; Newton-Raphson iteration with 2&times;2 numerical Jacobian (finite differences, &epsilon;=10&sup{&minus;8}). Converges to miss &lt;10&sup{&minus;11} AU.</div>
<div class="er" style="color:#ddd"><b>Gate corrections:</b> At each of 12 gates, a 2&times;4 sensitivity Jacobian J[k] maps state deviations to final miss. The velocity block Jv=J[:,2:4] solves &Delta;v = Jv&sup{&minus;1}&middot;(&minus;&Delta;miss). Damped by 0.7&times;(12&minus;k)/12 with |&Delta;v|&leq;0.05 AU/yr clamp.</div>
<h3 style="color:#ff8800">TARGET ORBITAL MOTION</h3>
<div class="er" style="color:#ddd"><b>Target prediction:</b> The target is not static &mdash; it follows its own Keplerian orbit. Its predicted path is computed via RK4 propagation from its initial state (position + orbital velocity). The intercept point is where the projectile meets the target&rsquo;s future position at time T<sub>f</sub>.</div>
<div class="er" style="color:#ddd"><b>Orbital velocity:</b> v<sub>tgt</sub> = &radic;(&mu;/r<sub>tgt</sub>) for circular orbit. For the default target at 1.52 AU: v &asymp; 4.14 AU/yr (&asymp; 19.6 km/s).</div>
<div class="er" style="color:#ddd"><b>Galactic motion:</b> Solar system barycentric velocity &asymp; 230 km/s (galactic rotation). Relative to Local Standard of Rest. This is a constant offset that cancels in the heliocentric frame used here, but becomes relevant for extra-solar targets.</div>
<h3 style="color:#66aaff">ENERGY &amp; MASS MODEL</h3>
<div class="er" style="color:#ddd"><b>Per-barrel projectile (default barrel knowledge):</b> every barrel &mdash; PRIMARY, PARALLEL, each SWARM node &mdash; ships with a projectile spec (diameter d, density &rho;, specific-impulse I<sub>sp</sub>). Mass is a sphere of uniform density: m = &rho;&sdot;&pi;d&sup3;/6. From that we derive the <b>power needed to achieve a hit</b> as the target or barrel placement changes:</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">KE = &frac12; m |v<sub>0</sub>|&sup2; &nbsp;|&nbsp; &Delta;GPE = GM&odot;&sdot;m&sdot;(1/r<sub>b</sub> &minus; 1/r<sub>t</sub>) &nbsp;|&nbsp; E<sub>tot</sub> = KE + max(&Delta;GPE,0)</div>
<div class="er" style="color:#ccaa66;font-family:monospace;font-size:10px">P<sub>avg</sub> = E<sub>tot</sub>/T<sub>f</sub> &nbsp;|&nbsp; &Delta;v<sub>tot</sub> = |v<sub>0</sub>| + &Sigma;|&Delta;v<sub>k</sub>| &nbsp;|&nbsp; m<sub>0</sub>/m<sub>f</sub> = exp(&Delta;v<sub>tot</sub>/(I<sub>sp</sub>&sdot;g<sub>0</sub>))</div>
<div class="er" style="color:#ddd"><b>Surfaces:</b> the CONFIG tab's &quot;PROJECTILE &mdash; default barrel knowledge&quot; block edits these inputs; the SCOPE HUD and PARALLEL tab show live KE / TNT equivalent / total E / average power / &Delta;v / fuel fraction; the SALVO info panel displays them per barrel so you can see the power-budget per firing position.</div>
<div class="er" style="color:#ddd"><b>Kinetic energy:</b> KE = &frac12;mv&sup2;. For a projectile of mass m at launch speed |v<sub>0</sub>|, the barrel must supply this energy plus gravitational potential energy escape cost.</div>
<div class="er" style="color:#ddd"><b>&Delta;v budget:</b> Total &Delta;v = |v<sub>0</sub>| + &Sigma;|&Delta;v<sub>k</sub>| (launch + 12 mid-course corrections). The Tsiolkovsky equation: &Delta;v = v<sub>e</sub> ln(m<sub>0</sub>/m<sub>f</sub>) constrains fuel mass fraction.</div>
<div class="er" style="color:#ddd"><b>Mass dependency:</b> Energy scales linearly with mass (E=mv&sup2;/2). A 10 km comet (&asymp;10&sup{12} kg) at 5.89 AU/yr requires &asymp;1.7&times;10&sup{29} J. Actual energy requirement depends on comet mass, distance, galactic motion offsets, and gravitational assists available.</div>
<div class="er" style="color:#ddd"><b>Variables affecting flow:</b> Solar radiation pressure, Yarkovsky effect (thermal re-emission), outgassing (for comets), relativistic corrections (negligible at these speeds), Jupiter perturbation, and local interstellar medium drag.</div>
<div class="er" style="color:#ddd;border:1px solid #ff8800;padding:6px;border-radius:4px"><b>NOTE:</b> Current barrel/target positions are <b>configurable placeholders</b>. Use Tab 6 (CONFIG) to set any barrel position, target location, mass, or flight parameters. The physics engine recalculates all trajectories, corrections, and energy budgets from the input parameters.</div>
<div class="ec">Press ESC to close &mdash; v5.16</div>
</div></div>
<script>
// ═══ LOADING SPLASH helper ═══
// Call ldStep(n, 'ok') to mark step n done, or ldStep(n, 'cur') to mark it
// in-progress. ldDone() dismisses the whole overlay. Updates the progress
// bar (#ldFill) + percent readout (#ldPct) based on step number.
let _LD_STEPS=5;
function ldStep(n,state){try{const el=document.getElementById('ls'+n);
if(el){el.className='s '+(state||'ok');
// replace trailing ·/· with done/cur glyph
const tick=state==='ok'?'✓':(state==='cur'?'…':'·');
const sp=el.querySelector('span');if(sp)sp.textContent=tick}
const done=Math.min(n,_LD_STEPS);const pct=Math.round(done/_LD_STEPS*100);
const f=document.getElementById('ldFill');if(f)f.style.width=pct+'%';
const p=document.getElementById('ldPct');if(p)p.textContent=pct+'%'}catch(e){}}
function ldDone(){try{const o=document.getElementById('load');if(o){
 ldStep(_LD_STEPS,'ok');
 setTimeout(()=>{o.classList.add('done');setTimeout(()=>{o.style.display='none'},400)},120)}}catch(e){}}
// Mark step 1 done — payload has parsed since we reached this line.
const D=/*__DATA__*/null;
ldStep(1,'ok');
const{B,TG,gxy:GXY,gt:GT,nt:NT,Rs,Rs_base:RS_BASE,hr:HR,res:R,bi:BI,ci:CI,T:TM,log:LOG,gi:GI,v0:V0,tf:TF,dtg:DTG,par:PAR,swm:SWM,tgt_orb:TGT_ORB,ener:ENER,
sun_pos:SUN_POS,sun_z:SUN_Z,tz:TGT_Z,use_perturbations:USE_PERT,solar:SOLAR}=D;
document.getElementById('hbh').textContent=R.bh+'%';
document.getElementById('hch').textContent=R.ch+'%';
document.getElementById('him').textContent=(R.ch-R.bh).toFixed(1)+'pts';
document.getElementById('hv0').textContent=Math.sqrt(V0[0]*V0[0]+V0[1]*V0[1]).toFixed(3)+' AU/yr';
document.getElementById('hfl').textContent=TF+' yr';
document.getElementById('hms').textContent=R.cmm+' AU';
document.getElementById('clog').textContent=LOG.join('\n');
// Scope scale readout in the title bar: AU radius / diameter / grow factor.
// The stencil is a scale unit — identical geometry at any diameter. When a target
// lands outside the base scope (Rs_base), Rs grows to keep the target on-stencil,
// letting one flower-of-life template cover solar → galactic → universal spans.
{const diam=(2*Rs).toFixed(3);const grow=(Rs/(RS_BASE||1.2));
const scopeStr='⭕ Scope Rs '+Rs.toFixed(3)+' AU · ⌀ '+diam+' AU · '+grow.toFixed(2)+'× base';
const hsEl=document.getElementById('hScope');if(hsEl)hsEl.textContent=scopeStr}

let hPts=[],rx=-0.4,ry=0.6,z3=1,drg=false,lmx,lmy;
let animT=0,playing=false,animSpd=1.0,mouseAU=[0,0];
let showSolar=true;  // toggle planets + moons rendering

// ═══════════════════════════════════════════════════════════════════════
// SHARED solar-system helpers (reachable from every tab's draw function).
//   solarAt(t)           → [ {idx,parent,x,y,name,color,r_vis,mu,kind,r_orbit,period} ]
//   solarHitScan(mx,my,toScreen)  → index into solarAt(t) of closest body under the cursor
//   SOLAR and SUN_POS are destructured above from the viz() payload.
// ═══════════════════════════════════════════════════════════════════════
function solarAt(t){
if(!SOLAR||!SOLAR.length)return[];
const out=[];const sx=SUN_POS?SUN_POS[0]:0,sy=SUN_POS?SUN_POS[1]:0;
for(let i=0;i<SOLAR.length;i++){const b=SOLAR[i];
 let x,y;
 if(i===0){x=sx;y=sy}
 else{const th=b.phase+2*Math.PI*t/Math.max(b.period,1e-12);
  x=sx+b.r_orbit*Math.cos(th);y=sy+b.r_orbit*Math.sin(th)}
 out.push({idx:i,parent:-1,x:x,y:y,name:b.name,color:b.color,r_vis:b.r_vis,mu:b.mu,kind:i===0?'sun':'planet',r_orbit:b.r_orbit,period:b.period,mi:-1});
 if(i===0||!b.moons)continue;
 for(let j=0;j<b.moons.length;j++){const m=b.moons[j];
  const mth=m.phase+2*Math.PI*t/Math.max(m.period,1e-12);
  const mx=x+m.r_orbit*Math.cos(mth);const my=y+m.r_orbit*Math.sin(mth);
  out.push({idx:i,parent:i,x:mx,y:my,name:m.name,color:m.color,r_vis:m.r_vis,mu:m.mu,kind:'moon',r_orbit:m.r_orbit,period:m.period,mi:j})}}
return out}

// Draw the solar system (sun + planets + moons + dashed orbits) via a caller-
// supplied `toScreen(x,y)` transform. Returns a list of screen hits so the
// caller can wire click-to-select and hover-tooltip behaviour. Shared across
// tabs 1, 3, 7, 8, 11, 12.
function drawSolarSystem2D(ctx,toScreen,opts){
if(!showSolar||!SOLAR||!SOLAR.length)return[];
const o=opts||{};
const scale=o.scale||1; // world-AU per pixel helper (used for visual radii)
const t=animT;
const bodies=solarAt(t);
const hits=[];
const sx=SUN_POS?SUN_POS[0]:0,sy=SUN_POS?SUN_POS[1]:0;
// Draw dashed orbits of planets (and moons if zoomed-in enough)
ctx.save();
if(o.drawOrbits!==false){
 for(let i=1;i<SOLAR.length;i++){const b=SOLAR[i];
  ctx.beginPath();ctx.strokeStyle=b.color+'44';ctx.lineWidth=0.9;ctx.setLineDash([3,4]);
  for(let k=0;k<=64;k++){const a=k*Math.PI*2/64;
   const p=toScreen(sx+b.r_orbit*Math.cos(a),sy+b.r_orbit*Math.sin(a));
   if(k===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}ctx.stroke()}
 ctx.setLineDash([])}
// Draw bodies
for(const bd of bodies){
 const p=toScreen(bd.x,bd.y);
 let r=Math.max(2,Math.min(14,(bd.r_vis||0.02)*scale));
 if(bd.kind==='sun'){ctx.save();ctx.shadowColor='#ffcc00';ctx.shadowBlur=18;
  ctx.beginPath();ctx.arc(p[0],p[1],Math.max(5,r),0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
  ctx.font='bold 11px sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('☼ Sun',p[0],p[1]-r-4);
  hits.push({x:p[0],y:p[1],r:Math.max(8,r+4),planet_idx:0,moon_idx:-1,body:bd})}
 else if(bd.kind==='planet'){ctx.beginPath();ctx.arc(p[0],p[1],r,0,Math.PI*2);ctx.fillStyle=bd.color;ctx.fill();
  ctx.strokeStyle='rgba(255,255,255,.5)';ctx.lineWidth=1;ctx.stroke();
  if(o.labels!==false){ctx.font='9px monospace';ctx.fillStyle=bd.color;ctx.textAlign='left';ctx.fillText(bd.name,p[0]+r+2,p[1]+3)}
  hits.push({x:p[0],y:p[1],r:Math.max(8,r+4),planet_idx:bd.idx,moon_idx:-1,body:bd})}
 else if(bd.kind==='moon'){ctx.beginPath();ctx.arc(p[0],p[1],Math.max(1.5,r*0.6),0,Math.PI*2);ctx.fillStyle=bd.color+'cc';ctx.fill();
  hits.push({x:p[0],y:p[1],r:Math.max(5,r+2),planet_idx:bd.parent,moon_idx:bd.mi,body:bd})}}
ctx.restore();
return hits}
// Sphere-nav address for ANY (x,y,z) world-AU — used by every hover tooltip.
// ═══════════════════════════════════════════════════════════════════════
// Tri-axis 360° sphere address. The navigation sphere is a 3D volume, and
// a single 360° angle can only describe points in one plane. So we emit
// THREE independent 360° rotations — one about each coordinate axis — plus
// the 3-axis grid offsets.
//
//   ax, ay, az    world AU (z defaults to 0 for 2D points)
//   gx, gy, gz    grid coords — Rs maps to ±30 per axis
//
//   Z-axis φ_Z = atan2(gy, gx)  ∈ [0°, 360°)   rotation in the XY plane
//   Y-axis φ_Y = atan2(gz, gx)  ∈ [0°, 360°)   rotation in the XZ plane
//   X-axis φ_X = atan2(gz, gy)  ∈ [0°, 360°)   rotation in the YZ plane
//   elevation θ = atan2(gz, √(gx²+gy²)) ∈ [-90°, +90°]  (kept for plots)
//
// addr:  "Zφ=201° Yφ=0° Xφ=0°, -12.9z, -4.9x, +0.0y"
//   — three axis rotations fully orient a direction in 3D; the ±gx/gy/gz
//   trio give the scale-normalised position on the 30-unit grid.
//
// addr2d / deg / elevation / azimuth kept for back-compatibility with code
// that read the old dual-angle fields.
// ═══════════════════════════════════════════════════════════════════════
function sphereAddr(ax,ay,az){
if(az===undefined||az===null)az=0;
const inv=1/Math.max(Rs,1e-9);
const gx=ax*inv*30,gy=ay*inv*30,gz=az*inv*30;
const rxy=Math.sqrt(gx*gx+gy*gy);
const r=Math.sqrt(gx*gx+gy*gy+gz*gz);
const wrap=(a)=>((a%360)+360)%360;
const phiZ=wrap(Math.atan2(gy,gx)*180/Math.PI);   // rotation about Z (XY plane)
const phiY=wrap(Math.atan2(gz,gx)*180/Math.PI);   // rotation about Y (XZ plane)
const phiX=wrap(Math.atan2(gz,gy)*180/Math.PI);   // rotation about X (YZ plane)
const elevation=Math.atan2(gz,rxy)*180/Math.PI;    // [-90, +90]
const frac=TF>0?Math.max(0,Math.min(1,animT/TF)):0;const cyc=frac*24;
const hh=Math.floor(cyc),mm=Math.floor((cyc-hh)*60),ss=Math.floor(((cyc-hh)*60-mm)*60);
const fmt=(v)=>((v>=0?'+':'')+v.toFixed(1));
const addr='Z'+Math.round(phiZ)+'° Y'+Math.round(phiY)+'° X'+Math.round(phiX)+'°, '+fmt(gx)+'z,'+fmt(gy)+'x,'+fmt(gz)+'y';
return{gx:gx,gy:gy,gz:gz,r:r,rxy:rxy,
deg:phiZ,azimuth:phiZ,elevation:elevation,
phiZ:phiZ,phiY:phiY,phiX:phiX,
addr:addr,
addr2d:Math.round(phiZ)+'°,'+gx.toFixed(1)+'z,'+gy.toFixed(1)+'x,0y',
time:String(hh).padStart(2,'0')+':'+String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0'),
inScope:r<=30}}
let showParallel=true,showSwarm=true;
let salvoFocus=-1; // -1=show all, 0+=focused barrel index — GLOBAL: used by ALL tabs to orient at 7 o'clock
let s3rx=-0.3,s3ry=0.5,s3z=1,s3drg=false; // 3D Sphere tab rotation state
let p3rx=-0.4,p3ry=0.6,p3z=1,p3drg=false,p3lmx=0,p3lmy=0; // PARALLEL 3D tab rotation state
// === Simulation.py camera model: world-coord pan + zoom-toward-mouse ===
let camX=0,camY=0,camZ=1;   // camX,camY in AU world-coords; camZ = zoom multiplier
const MIN_ZOOM=0.005,MAX_ZOOM=200;  // galactic to intergalactic scales
let panDrg=false;
// === Display settings: font scale + layer visibility ===
let FS=1.0; // font scale multiplier (0.2-1.5, default 1.0)
const DV={L:true,G:true,O:true,T:true,X:true,Tgt:true,Bar:true,Par:true,Swm:true,Sph:true,En:true,Grid:true};
function fs(basePx){return Math.round(basePx*FS)+'px'}
// Display panel toggle
document.getElementById('hkDsp').onclick=()=>document.getElementById('dsp').classList.toggle('on');
// Font slider
document.getElementById('fsSlider').oninput=function(){FS=this.value/100;document.getElementById('fsVal').textContent=FS.toFixed(2)+'x';draw()};
// Layer checkboxes
['dL','dG','dO','dT','dX','dTgt','dBar','dPar','dSwm','dSph','dEn','dGrid'].forEach(id=>{
document.getElementById(id).onchange=function(){const k=id.replace('d','');DV[k]=this.checked;
if(k==='Par')showParallel=this.checked;else if(k==='Swm')showSwarm=this.checked;
if(typeof syncLegend==='function')syncLegend();draw()}});
function dspAll(v){['dL','dG','dO','dT','dX','dTgt','dBar','dPar','dSwm','dSph','dEn','dGrid'].forEach(id=>{
document.getElementById(id).checked=v;const k=id.replace('d','');DV[k]=v});syncLegend();draw()}

// ═══════════════════════════════════════════════════════════════════════
// Shared clickable LEGEND. Each row has a colored swatch + real checkbox
// bound to either a DV flag or a scalar toggle (showParallel / showSwarm).
// Clicking the row toggles the checkbox and redraws. This replaces the
// read-only canvas-drawn legends that used to live inside each tab.
// ═══════════════════════════════════════════════════════════════════════
const LEGEND_ROWS=[
{sw:'#ffcc00',lbl:'◇ FTOP (origin)',flag:null},
{sw:'rgba(120,170,255,.9)',lbl:'Reference orbits',flag:'O'},
{sw:'#ffd700',lbl:'Correction gates',flag:'G'},
{sw:'#44ff88',lbl:'Trajectory (speed)',flag:'T'},
{sw:'rgba(220,185,120,.9)',lbl:'Scope / FoL lattice',flag:'X'},
{sw:'#4488ff',lbl:'Barrel / pusher',flag:'Bar'},
{sw:'rgba(255,170,60,.85)',lbl:'Fibonacci sections',flag:'X'},
{sw:'#44ff88',lbl:'Target + hit radius',flag:'Tgt'},
{sw:'#00dd88',lbl:'Parallel scope',flag:'P'},
{sw:'#cc44ff',lbl:'Swarm barrels',flag:'S'},
{sw:'rgba(255,215,0,.7)',lbl:'Labels & text',flag:'L'},
{sw:'rgba(150,135,105,.7)',lbl:'Grid lines',flag:'Grid'}];
function buildLegend(){const body=document.getElementById('lgBody');if(!body)return;
let h='';for(const r of LEGEND_ROWS){
const chkId='lg_'+(r.flag||'FTOP');
const checked=r.flag===null?true:(r.flag==='P'?showParallel:(r.flag==='S'?showSwarm:DV[r.flag]));
h+='<label class="lgrow"><input type="checkbox" id="'+chkId+'"'+(checked?' checked':'')+(r.flag===null?' disabled':'')+'>'+
'<span class="lgsw" style="background:'+r.sw+'"></span><span class="lgtxt">'+r.lbl+'</span></label>'}
body.innerHTML=h;
// Wire handlers
for(const r of LEGEND_ROWS){if(r.flag===null)continue;
const el=document.getElementById('lg_'+r.flag);if(!el)continue;
el.onchange=function(){
if(r.flag==='P'){showParallel=this.checked;DV.Par=this.checked;const d=document.getElementById('dPar');if(d)d.checked=this.checked}
else if(r.flag==='S'){showSwarm=this.checked;DV.Swm=this.checked;const d=document.getElementById('dSwm');if(d)d.checked=this.checked}
else{DV[r.flag]=this.checked;const dsp=document.getElementById('d'+r.flag);if(dsp)dsp.checked=this.checked}
syncLegend();draw()}}}
// Keep legend checkboxes in sync with DV / showParallel / showSwarm (called
// whenever those change via keyboard shortcuts or the Display panel).
function syncLegend(){for(const r of LEGEND_ROWS){if(r.flag===null)continue;
const el=document.getElementById('lg_'+r.flag);if(!el)continue;
el.checked=(r.flag==='P'?showParallel:(r.flag==='S'?showSwarm:DV[r.flag]))}}
// Collapse toggle
{const tog=document.getElementById('lgTog');const lg=document.getElementById('legend');
if(tog&&lg){const h4=lg.querySelector('h4');if(h4)h4.onclick=()=>{lg.classList.toggle('collapsed');tog.textContent=lg.classList.contains('collapsed')?'[+]':'[−]'}}}
// Adaptive scale label: returns human-readable unit for current zoom
function scaleUnit(au){if(au>1e4)return[(au/206265).toFixed(2)+' pc',206265];if(au>1e3)return[(au/1000).toFixed(1)+'k AU',1];if(au>10)return[au.toFixed(0)+' AU',1];if(au>0.1)return[au.toFixed(2)+' AU',1];return[(au*1.496e8).toFixed(0)+' km',1]}
const MU=4*Math.PI*Math.PI;const DAMP=0.7;const MAX_DV=0.05;
// Fibonacci gate indices for labeling (gates 1-12, pick Fib subset).
// Declared here so STATS and other early blocks can reference it.
const SAL_FIB=new Set([1,2,3,5,8]);
const SWM_COLS=['#00cc88','#cc44ff','#ff8844','#44aaff','#ff4488'];
const tabs=document.querySelectorAll('.tb'),pgs=document.querySelectorAll('.pg');
function switchTab(i){tabs.forEach(t=>t.classList.remove('on'));pgs.forEach(p=>p.classList.remove('on'));
tabs[i].classList.add('on');document.getElementById(tabs[i].dataset.t).classList.add('on');requestAnimationFrame(draw)}
tabs.forEach((b,i)=>{b.onclick=()=>switchTab(i)});

// Animation helpers
function getAnimPos(){const f=animT/TF;const idx=f*(NT.length-1);const i0=Math.floor(idx),i1=Math.min(i0+1,NT.length-1);const tt=idx-i0;
return[NT[i0][0]+(NT[i1][0]-NT[i0][0])*tt,NT[i0][1]+(NT[i1][1]-NT[i0][1])*tt]}
function getAnimSpd(){const gIdx=Math.min(Math.floor(animT/DTG),11);const g=GI[gIdx];return g?g.spd:GI[11].spd}
function getAnimGate(){return Math.min(Math.floor(animT/DTG)+1,12)}
function updateCtrl(){document.getElementById('tDisp').textContent=animT.toFixed(3);
document.getElementById('sDisp').textContent=getAnimSpd().toFixed(2);
document.getElementById('gDisp').textContent=getAnimGate()+'/12';
document.getElementById('timeline').value=Math.round(animT/TF*1000);
document.getElementById('playBtn').textContent=playing?'\u23f8 PAUSE':'\u25b6 PLAY';
document.getElementById('playBtn').className=playing?'act':''}
// ---------- shared live-kinematics helper ----------
// Build the same math dossier used by the SCOPE hud AND the PARALLEL tab hud,
// so both panels stay coherent and complete.
function liveKinematics(p,v,animTL){
const r=Math.sqrt(p[0]*p[0]+p[1]*p[1])||1e-12;
const spd=Math.sqrt(v[0]*v[0]+v[1]*v[1]);
const ax=-MU*p[0]/(r*r*r),ay=-MU*p[1]/(r*r*r);
const ag=Math.sqrt(ax*ax+ay*ay);
// radial/tangential split
const rHatX=p[0]/r,rHatY=p[1]/r;
const vr=v[0]*rHatX+v[1]*rHatY;          // dr/dt (radial speed, signed)
const vt=v[0]*(-rHatY)+v[1]*rHatX;       // tangential speed, signed
// angular momentum (specific): L = x*vy - y*vx
const L=p[0]*v[1]-p[1]*v[0];
// specific orbital energy, vis-viva
const eps=0.5*spd*spd-MU/r;
// semi-major axis, period, eccentricity
const sma=(Math.abs(eps)<1e-12)?Infinity:(-MU/(2*eps));
const period=(sma>0)?2*Math.PI*Math.sqrt(sma*sma*sma/MU):null;
// eccentricity vector (planar): e = (v\u00d7L_z)/mu - r/|r|  ; here L_z is scalar L
const evx=( v[1]*L)/MU-rHatX;
const evy=(-v[0]*L)/MU-rHatY;
const ecc=Math.sqrt(evx*evx+evy*evy);
// flight-path angle: angle between velocity and local perpendicular-to-radius
const gamma=Math.atan2(vr,Math.abs(vt))*180/Math.PI;
// escape velocity here
const vesc=Math.sqrt(2*MU/r);
// jerk magnitude from 2-body gravity: dA/dt = -MU * (v/r^3 - 3 r (r\u00b7v)/r^5)
const rv=p[0]*v[0]+p[1]*v[1];
const jerkX=-MU*(v[0]/(r*r*r)-3*p[0]*rv/(r*r*r*r*r));
const jerkY=-MU*(v[1]/(r*r*r)-3*p[1]*rv/(r*r*r*r*r));
const jerk=Math.sqrt(jerkX*jerkX+jerkY*jerkY);
// === extended orbital element set ===
// argument of periapsis (\u03c9) \u2014 angle of eccentricity vector from +x
const omega=(ecc>1e-6)?(Math.atan2(evy,evx)*180/Math.PI):null;
// true anomaly (\u03bd) \u2014 angle from periapsis to current position
const nu=(ecc>1e-6)?(Math.atan2(p[0]*evy-p[1]*evx,p[0]*evx+p[1]*evy)*180/Math.PI):(Math.atan2(p[1],p[0])*180/Math.PI);
// periapsis + apoapsis radii from (a, e)
const rp=(isFinite(sma)&&sma>0)?sma*(1-ecc):null;
const ra=(isFinite(sma)&&sma>0&&ecc<1)?sma*(1+ecc):null;
// mean motion n = sqrt(\u03bc/a\u00b3) rad/yr
const nMean=(isFinite(sma)&&sma>0)?Math.sqrt(MU/(sma*sma*sma)):null;
// specific angular-momentum magnitude (scalar L already) vs semi-latus rectum p_slr
const p_slr=L*L/MU;
// vis-viva check: v\u00b2 = \u03bc(2/r \u2212 1/a)  \u2192 return residual so user can see numerical sanity
const visViva=(isFinite(sma)&&sma!==0)?(spd*spd-MU*(2/r-1/sma)):0;
// gravity-gradient / tidal tensor eigenvalue (radial): Uxx = -\u03bc/r\u00b3 + 3\u03bcx\u00b2/r\u2075
const r3=r*r*r,r5=r3*r*r;
const Uxx=-MU/r3+3*MU*p[0]*p[0]/r5;
const Uyy=-MU/r3+3*MU*p[1]*p[1]/r5;
const Uxy=3*MU*p[0]*p[1]/r5;
const tidalTr=Uxx+Uyy;                    // 0 in 2-body (Laplace: \u2207\u00b2U = 0 away from mass)
const tidalDet=Uxx*Uyy-Uxy*Uxy;
// relativistic factors
const C_AUYR=63241.1;
const beta=spd/C_AUYR;
const gam=1/Math.sqrt(Math.max(1e-30,1-beta*beta));
// solar-radiation pressure acceleration on a 100m / 2600 kg/m\u00b3 body (SI \u2192 AU/yr\u00b2)
const AU_M=1.496e11,YR_S=3.156e7;
const spd_kms=spd*AU_M/YR_S/1000.0;
const ag_mms2=ag*AU_M/YR_S/YR_S*1000.0;
// ETA to target along straight-line approximation
const tRemain=Math.max(0,TF-animTL);
return{r:r,spd:spd,ax:ax,ay:ay,ag:ag,vr:vr,vt:vt,L:L,eps:eps,sma:sma,
period:period,ecc:ecc,gamma:gamma,vesc:vesc,jerk:jerk,
spd_kms:spd_kms,ag_mms2:ag_mms2,tRemain:tRemain,
omega:omega,nu:nu,rp:rp,ra:ra,nMean:nMean,p_slr:p_slr,visViva:visViva,
Uxx:Uxx,Uyy:Uyy,Uxy:Uxy,tidalTr:tidalTr,tidalDet:tidalDet,
beta:beta,gam:gam,rv:rv}}

function updateHud(){const p=getAnimPos();const g=getAnimGate();const gi=GI[Math.min(g-1,11)];
const spd=getAnimSpd();
// Compute live derivatives
const vx=gi?gi.vx:V0[0],vy=gi?gi.vy:V0[1];
const km=liveKinematics(p,[vx,vy],animT);
const r=km.r;
const dTgt=Math.sqrt((p[0]-TG[0])**2+(p[1]-TG[1])**2);
const dBar=Math.sqrt((p[0]-B[0])**2+(p[1]-B[1])**2);
const totalD=Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2);
const pctComplete=(dBar/totalD*100).toFixed(1);
const kappa=gi?parseFloat(gi.curv):0;
let h='<span class="hd">LIVE TRAJECTORY DATA</span><br>';
h+='<span class="hd">T:</span> <span class="hl">+'+animT.toFixed(4)+' yr</span> ('+((animT/TF)*100).toFixed(1)+'%) | ETA '+km.tRemain.toFixed(3)+' yr<br>';
h+='<span class="hd">Pos:</span> <span class="hl">('+p[0].toFixed(4)+', '+p[1].toFixed(4)+')</span> AU<br>';
h+='<span class="hd">Speed:</span> <span class="hl">'+spd.toFixed(3)+'</span> AU/yr <span class="hd">(</span><span class="hl">'+km.spd_kms.toFixed(2)+' km/s</span><span class="hd">)</span><br>';
h+='<span class="hd">R(FTOP):</span> <span class="hl">'+r.toFixed(4)+'</span> AU<br>';
h+='<span class="hd">Gate:</span> <span class="hl">'+g+'/12</span>';
if(gi)h+=' | Hdg: '+gi.hdg+'\u00b0';
h+='<br><span style="color:#c8a96e;font:bold 8px monospace">\u2500\u2500 CALCULUS \u2500\u2500</span><br>';
h+='<span class="hd">dx/dt:</span> <span class="hl">'+vx.toFixed(4)+'</span> AU/yr<br>';
h+='<span class="hd">dy/dt:</span> <span class="hl">'+vy.toFixed(4)+'</span> AU/yr<br>';
h+='<span class="hd">v<sub>r</sub> (radial):</span> <span class="hl">'+km.vr.toFixed(4)+'</span> AU/yr<br>';
h+='<span class="hd">v<sub>t</sub> (tangential):</span> <span class="hl">'+km.vt.toFixed(4)+'</span> AU/yr<br>';
h+='<span class="hd">d\u00b2x/dt\u00b2:</span> <span class="hl">'+km.ax.toFixed(4)+'</span> AU/yr\u00b2<br>';
h+='<span class="hd">d\u00b2y/dt\u00b2:</span> <span class="hl">'+km.ay.toFixed(4)+'</span> AU/yr\u00b2<br>';
h+='<span class="hd">|a|=\u03bc/r\u00b2:</span> <span class="hl">'+km.ag.toFixed(4)+'</span> AU/yr\u00b2 <span class="hd">(</span><span class="hl">'+km.ag_mms2.toFixed(3)+' mm/s\u00b2</span><span class="hd">)</span><br>';
h+='<span class="hd">|jerk|:</span> <span class="hl">'+km.jerk.toFixed(4)+'</span> AU/yr\u00b3<br>';
h+='<span class="hd">\u03ba(s):</span> <span class="hl">'+kappa.toFixed(4)+'</span> /AU<br>';
h+='<span style="color:#c8a96e;font:bold 8px monospace">\u2500\u2500 ORBIT MECHANICS \u2500\u2500</span><br>';
h+='<span class="hd">\u03b5 (spec. energy):</span> <span class="hl">'+km.eps.toFixed(4)+'</span> AU\u00b2/yr\u00b2<br>';
h+='<span class="hd">L (ang. mom.):</span> <span class="hl">'+km.L.toFixed(5)+'</span> AU\u00b2/yr<br>';
h+='<span class="hd">p = L\u00b2/\u03bc:</span> <span class="hl">'+km.p_slr.toFixed(5)+'</span> AU<br>';
h+='<span class="hd">a (semi-major):</span> <span class="hl">'+(isFinite(km.sma)?km.sma.toFixed(4):'\u221e')+'</span> AU<br>';
h+='<span class="hd">e (eccentricity):</span> <span class="hl">'+km.ecc.toFixed(4)+'</span><br>';
if(km.omega!=null)h+='<span class="hd">\u03c9 (arg peri.):</span> <span class="hl">'+km.omega.toFixed(2)+'\u00b0</span><br>';
h+='<span class="hd">\u03bd (true anom.):</span> <span class="hl">'+km.nu.toFixed(2)+'\u00b0</span><br>';
if(km.rp!=null)h+='<span class="hd">r<sub>p</sub> periapsis:</span> <span class="hl">'+km.rp.toFixed(4)+'</span> AU<br>';
if(km.ra!=null)h+='<span class="hd">r<sub>a</sub> apoapsis:</span> <span class="hl">'+km.ra.toFixed(4)+'</span> AU<br>';
if(km.period!=null)h+='<span class="hd">T (period):</span> <span class="hl">'+km.period.toFixed(4)+'</span> yr<br>';
if(km.nMean!=null)h+='<span class="hd">n (mean motion):</span> <span class="hl">'+km.nMean.toFixed(5)+'</span> rad/yr<br>';
h+='<span class="hd">\u03b3 (flight-path):</span> <span class="hl">'+km.gamma.toFixed(2)+'\u00b0</span><br>';
h+='<span class="hd">v<sub>esc</sub>:</span> <span class="hl">'+km.vesc.toFixed(4)+'</span> AU/yr <span class="hd">(v/v<sub>esc</sub>=</span><span class="hl">'+(km.spd/km.vesc).toFixed(3)+'</span><span class="hd">)</span><br>';
h+='<span class="hd">vis-viva residual:</span> <span class="hl">'+km.visViva.toExponential(2)+'</span> AU\u00b2/yr\u00b2 (\u21920 is exact)<br>';
h+='<span class="hd">r\u00b7v:</span> <span class="hl">'+km.rv.toFixed(4)+'</span> AU\u00b2/yr (sign: +outbound / \u2212inbound)<br>';
h+='<span style="color:#c8a96e;font:bold 8px monospace">\u2500\u2500 TIDAL / GRAV-GRAD TENSOR \u2500\u2500</span><br>';
h+='<span class="hd">U<sub>xx</sub>:</span> <span class="hl">'+km.Uxx.toFixed(4)+'</span> /yr\u00b2<br>';
h+='<span class="hd">U<sub>yy</sub>:</span> <span class="hl">'+km.Uyy.toFixed(4)+'</span> /yr\u00b2<br>';
h+='<span class="hd">U<sub>xy</sub>:</span> <span class="hl">'+km.Uxy.toFixed(4)+'</span> /yr\u00b2<br>';
h+='<span class="hd">tr(\u2207\u2207U):</span> <span class="hl">'+km.tidalTr.toExponential(2)+'</span> (Laplace\u22480)<br>';
h+='<span class="hd">det(\u2207\u2207U):</span> <span class="hl">'+km.tidalDet.toExponential(2)+'</span><br>';
h+='<span style="color:#c8a96e;font:bold 8px monospace">\u2500\u2500 RELATIVISTIC \u2500\u2500</span><br>';
h+='<span class="hd">\u03b2 = v/c:</span> <span class="hl">'+km.beta.toExponential(3)+'</span><br>';
h+='<span class="hd">\u03b3 (Lorentz):</span> <span class="hl">'+km.gam.toFixed(12)+'</span><br>';
h+='<span class="hd">\u03c4/T:</span> <span class="hl">'+(1/km.gam).toFixed(12)+'</span> (proper-time factor)<br>';
// Gate-level numerics (from the current gate index, fed by GI)
if(gi){h+='<span style="color:#c8a96e;font:bold 8px monospace">\u2500\u2500 GATE '+g+' NUMERICS \u2500\u2500</span><br>';
h+='<span class="hd">J-cond:</span> <span class="hl">'+gi.jc+'</span> (\u2016J\u2016\u00b7\u2016J\u207b\u00b9\u2016)<br>';
h+='<span class="hd">Arc:</span> <span class="hl">'+gi.arc+'</span> AU between gates<br>';
h+='<span class="hd">\u03ba (curvature):</span> <span class="hl">'+gi.curv+'</span> /AU<br>';
h+='<span class="hd">\u0394speed:</span> <span class="hl" style="color:'+(gi.dspd>=0?'#44ff88':'#ff8866')+'">'+(gi.dspd>=0?'+':'')+gi.dspd+'</span> AU/yr<br>';
h+='<span class="hd">\u0394hdg:</span> <span class="hl">'+(gi.dhdg>=0?'+':'')+gi.dhdg+'\u00b0</span><br>';
h+='<span class="hd">damping:</span> <span class="hl">'+(DAMP*Math.max(0.2,(12-(g-1))/12)).toFixed(3)+'</span> \u00d7 \u0394v (decays late)<br>';
h+='<span class="hd">\u2016\u0394v\u2016 limit:</span> <span class="hl">'+MAX_DV+'</span> AU/yr per gate<br>';
h+='<span class="hd">t<sub>go</sub>:</span> <span class="hl">'+gi.tr+'</span> yr of flight remaining<br>'}
h+='<span style="color:#c8a96e;font:bold 8px monospace">\u2500\u2500 GAP MATH \u2500\u2500</span><br>';
h+='<span class="hd">d(Barrel):</span> <span class="hl">'+dBar.toFixed(4)+'</span> AU<br>';
h+='<span class="hd">d(Target):</span> <span class="hl" style="color:#44ff88">'+dTgt.toFixed(4)+'</span> AU<br>';
h+='<span class="hd">Progress:</span> <span class="hl">'+pctComplete+'%</span> of '+totalD.toFixed(3)+' AU<br>';
// Impact clock (simple closing-speed ETA along the straight line)
const closingSpd=km.vr<0?Math.abs(km.vr):km.spd;
const impactETA=(closingSpd>1e-6)?(dTgt/closingSpd):Infinity;
h+='<span class="hd">Impact ETA (est):</span> <span class="hl">'+(isFinite(impactETA)?impactETA.toFixed(4):'\u2014')+'</span> yr';
// Parallel scope diff (always show to make tab 1 and tab 11 coherent)
if(PAR){const gk=Math.min(g-1,11);const d=PAR.diff[gk];
h+='<br><span style="color:#00dd88;font:bold 8px monospace">\u2500\u2500 PARALLEL \u0394 \u2500\u2500</span><br>';
h+='<span class="hd">\u0394pos:</span> <span class="hl" style="color:#00dd88">'+d.d_pos+' AU</span><br>';
h+='<span class="hd">\u0394vel:</span> <span class="hl" style="color:#00dd88">'+d.d_vel+' AU/yr</span><br>';
h+='<span class="hd">\u0394hdg:</span> <span class="hl" style="color:#00dd88">'+d.d_hdg+'\u00b0</span>'}
// Energy / power needed for this barrel's default projectile
if(ENER&&ENER.primary_proj){const pr=ENER.primary_proj;
h+='<br><span style="color:#ffd700;font:bold 8px monospace">\u2500\u2500 PROJECTILE &amp; ENERGY \u2500\u2500</span><br>';
h+='<span class="hd">\u2300:</span> <span class="hl">'+pr.diameter_m+' m</span> | <span class="hd">m:</span> <span class="hl">'+pr.mass_kg.toExponential(3)+' kg</span><br>';
h+='<span class="hd">KE:</span> <span class="hl">'+pr.ke_j.toExponential(3)+' J</span> ('+pr.tnt_kt.toFixed(2)+' kt TNT)<br>';
h+='<span class="hd">Total E:</span> <span class="hl">'+pr.total_j.toExponential(3)+' J</span><br>';
h+='<span class="hd">P<sub>avg</sub>:</span> <span class="hl">'+pr.power_avg_W.toExponential(3)+' W</span><br>';
h+='<span class="hd">\u0394v total:</span> <span class="hl">'+pr.dv_total_kms.toFixed(2)+' km/s</span> | <span class="hd">fuel:</span> <span class="hl">'+(pr.fuel_frac*100).toFixed(2)+'%</span>'}
// Swarm info
if(showSwarm&&SWM){h+='<br><span style="color:#cc44ff;font:bold 8px monospace">\u2500\u2500 SWARM \u2500\u2500</span><br>';
h+='<span class="hd">Barrels:</span> <span class="hl">'+SWM.length+'</span> \u2192 all converge on TARGET'}
document.getElementById('hud').innerHTML=h;
// The PARALLEL tab uses a sibling hud fed by updateParHud() below.
if(typeof updateParHud==='function')updateParHud()}

// Target panel
document.getElementById('tgtX').value=TG[0].toFixed(4);
document.getElementById('tgtY').value=TG[1].toFixed(4);
document.getElementById('tgtD').textContent=Math.sqrt(TG[0]*TG[0]+TG[1]*TG[1]).toFixed(4);
function toggleTgt(){document.getElementById('tpan').classList.toggle('on')}
function updateTgtDist(){const x=parseFloat(document.getElementById('tgtX').value)||0;
const y=parseFloat(document.getElementById('tgtY').value)||0;
document.getElementById('tgtD').textContent=Math.sqrt(x*x+y*y).toFixed(4)}
document.getElementById('tgtX').oninput=updateTgtDist;
document.getElementById('tgtY').oninput=updateTgtDist;
document.getElementById('tgtBtn').onclick=toggleTgt;
document.getElementById('tgtClose').onclick=toggleTgt;

// Control bar
document.getElementById('playBtn').onclick=()=>{playing=!playing;updateCtrl()};
document.getElementById('stepBk').onclick=()=>{animT=Math.max(0,animT-DTG);playing=false;updateCtrl();updateHud();draw()};
document.getElementById('stepFw').onclick=()=>{animT=Math.min(TF,animT+DTG);playing=false;updateCtrl();updateHud();draw()};
document.getElementById('timeline').oninput=e=>{animT=e.target.value/1000*TF;playing=false;updateCtrl();updateHud();draw()};
document.getElementById('hkZoomIn').onclick=()=>{camZ=Math.min(MAX_ZOOM,camZ*1.15);draw()};
document.getElementById('hkZoomOut').onclick=()=>{camZ=Math.max(MIN_ZOOM,camZ/1.15);draw()};
document.getElementById('hkReset').onclick=()=>{rx=-0.4;ry=0.6;z3=1;camX=0;camY=0;camZ=1;draw()};
document.getElementById('hkPar').onclick=()=>{showParallel=!showParallel;draw()};
document.getElementById('hkSwm').onclick=()=>{showSwarm=!showSwarm;draw()};
document.getElementById('hkEsc').onclick=()=>{document.getElementById('esc').classList.toggle('on')};

// Animation loop
function animLoop(){if(playing){animT+=0.003*animSpd;if(animT>=TF){animT=TF;playing=false}updateCtrl();updateHud();draw()}requestAnimationFrame(animLoop)}
requestAnimationFrame(animLoop);updateHud();
buildLegend();

// Precompute Flower-of-Life cross-section intersections — auto-scales to target distance
const XS=(function(){
const r1=Rs*0.58;const cc=[[0,0,r1]];
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;cc.push([r1*Math.cos(a),r1*Math.sin(a),r1])}
for(let i=0;i<12;i++){const a=i*Math.PI/6;cc.push([Rs*Math.cos(a),Rs*Math.sin(a),Rs])}
for(let i=0;i<12;i++){const a=(i*30+15)*Math.PI/180;cc.push([Rs*1.5*Math.cos(a),Rs*1.5*Math.sin(a),Rs])}
// ═══ AUTO-SCALE: extend flower toward target (and barrel) so scope covers full corridor ═══
const tgD=Math.sqrt(TG[0]*TG[0]+TG[1]*TG[1]);
const bD=Math.sqrt(B[0]*B[0]+B[1]*B[1]);
const maxD=Math.max(tgD,bD,Rs*1.5);
if(maxD>Rs*1.5){
const dirs=[[TG[0],TG[1]],[B[0],B[1]]];
for(const[dx,dy]of dirs){const dr=Math.sqrt(dx*dx+dy*dy)||1;
const ux=dx/dr,uy=dy/dr;
const nExt=Math.ceil(dr/(Rs*0.8));
for(let layer=1;layer<=nExt;layer++){
const dist=layer*Rs*0.9;if(dist>dr+Rs)break;
const cx2=ux*dist,cy2=uy*dist;
cc.push([cx2,cy2,r1]);
for(let i=0;i<6;i++){const a=i*Math.PI/3;
cc.push([cx2+r1*Math.cos(a),cy2+r1*Math.sin(a),r1])}}}}
function ci(ax,ay,ar,bx,by,br){const dx=bx-ax,dy=by-ay,d=Math.sqrt(dx*dx+dy*dy);
if(d>ar+br-0.001||d<Math.abs(ar-br)+0.001||d<0.01)return[];
const a2=(ar*ar-br*br+d*d)/(2*d),hh=ar*ar-a2*a2;if(hh<0.0001)return[];
const h2=Math.sqrt(hh),mx=ax+a2*dx/d,my=ay+a2*dy/d,px=h2*dy/d,py=h2*dx/d;
return[[mx+px,my-py],[mx-px,my+py]]}
const pts=[];
for(let i=0;i<cc.length;i++)for(let j=i+1;j<cc.length;j++){
const ps=ci(cc[i][0],cc[i][1],cc[i][2],cc[j][0],cc[j][1],cc[j][2]);
for(const p of ps){let dup=false;
for(const q of pts){if(Math.abs(p[0]-q[0])<0.02&&Math.abs(p[1]-q[1])<0.02){q[2]++;dup=true;break}}
if(!dup)pts.push([p[0],p[1],1])}}
// Sort spiral: by radius, then clockwise from 12 o'clock
pts.sort((a,b)=>{const ra=Math.sqrt(a[0]**2+a[1]**2),rb=Math.sqrt(b[0]**2+b[1]**2);
if(Math.abs(ra-rb)>0.04)return ra-rb;
const aa=(Math.PI/2-Math.atan2(a[1],a[0])+4*Math.PI)%(2*Math.PI);
const ab=(Math.PI/2-Math.atan2(b[1],b[0])+4*Math.PI)%(2*Math.PI);return aa-ab});
return pts})();
ldStep(2,'ok');
const FIB_SET=new Set([0,1,2,3,5,8,13,21,34,55,89]);
ldStep(3,'ok');

document.addEventListener('keydown',e=>{
if(e.target.tagName==='INPUT')return;
if(e.key==='Escape'){document.getElementById('esc').classList.toggle('on');return}
if(e.key===' '){e.preventDefault();playing=!playing;updateCtrl();return}
if(e.key==='t'||e.key==='T'){toggleTgt();return}
const n=parseInt(e.key);if(n>=1&&n<=9)switchTab(n-1);if(e.key==='0')switchTab(9);
if(e.key==='p'||e.key==='P'){showParallel=!showParallel;DV.Par=showParallel;const d=document.getElementById('dPar');if(d)d.checked=showParallel;if(typeof syncLegend==='function')syncLegend();draw();return}
if(e.key==='s'||e.key==='S'){showSwarm=!showSwarm;DV.Swm=showSwarm;const d=document.getElementById('dSwm');if(d)d.checked=showSwarm;if(typeof syncLegend==='function')syncLegend();draw();return}
if(e.key==='d'||e.key==='D'){document.getElementById('dsp').classList.toggle('on');return}
if(e.key==='r'||e.key==='R'){rx=-0.4;ry=0.6;z3=1;camX=0;camY=0;camZ=1;p3rx=-0.4;p3ry=0.6;p3z=1;draw()}
if(e.key==='+'||e.key==='='){camZ=Math.min(MAX_ZOOM,camZ*1.15);draw()}
if(e.key==='-'||e.key==='_'){camZ=Math.max(MIN_ZOOM,camZ/1.15);draw()}
if(e.key==='ArrowRight'){animT=Math.min(TF,animT+DTG);playing=false;updateCtrl();updateHud();draw()}
if(e.key==='ArrowLeft'){animT=Math.max(0,animT-DTG);playing=false;updateCtrl();updateHud();draw()}
});

function draw(){const a=document.querySelector('.pg.on');if(!a)return;
if(a.id==='sc-pg')drawScope();else if(a.id==='imp-pg')drawImpact();else if(a.id==='v3-pg')draw3D();else if(a.id==='sal-pg')drawSalvo();else if(a.id==='trj-pg')drawTrajectory();else if(a.id==='sph-pg')drawSphere();else if(a.id==='s3d-pg')draw3DSphere();else if(a.id==='par-pg')drawParallel();else if(a.id==='p3d-pg')drawParallel3D()}

function ts(x,y,cx,cy,sc){return[cx+x*sc,cy-y*sc]}
function circ(ctx,x,y,r,col,lw,dash){ctx.beginPath();ctx.arc(x,y,Math.max(0,r),0,Math.PI*2);ctx.strokeStyle=col;ctx.lineWidth=lw||1;ctx.setLineDash(dash||[]);ctx.stroke();ctx.setLineDash([])}

function drawScope(){
const c=document.getElementById('sC');const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
// === Simulation.py camera: world-coord center, zoom-toward-mouse ===
// baseSc is normalised by Rs/Rs_base so the scope template always occupies
// the same screen fraction regardless of scope scale — enlarging Rs is a
// scroll-out ("same look, more AU covered"), not a tile operation.
const baseSc=Math.min(w,h)/5.6*(RS_BASE/Math.max(Rs,1e-9));
const sc=baseSc*camZ;
const cx=w/2,cy=h/2;
// t() converts AU world coords → screen pixel coords
// ALL objects use this — target, barrel, scope, gates, Fibonacci — everything pans together
const t=(x,y)=>[cx+(x-camX)*sc,cy-(y-camY)*sc];
ctx.fillStyle='#0a0a1a';ctx.fillRect(0,0,w,h);
hPts=[];

// Orbit rings (gravitational influences) — brighter + thicker for contrast
let[ox,oy]=t(0,0);
if(DV.O){
circ(ctx,ox,oy,0.72*sc,'rgba(230,200,120,.75)',2.2,[7,5]);
circ(ctx,ox,oy,1.0*sc,'rgba(140,180,255,.8)',2.2,[7,5]);
circ(ctx,ox,oy,1.52*sc,'rgba(255,140,100,.75)',2.2,[7,5]);
if(DV.L){ctx.font='bold '+fs(28)+' sans-serif';ctx.textAlign='left';
let[lx,ly]=t(0.51,0.51);ctx.fillStyle='rgba(210,180,100,.7)';ctx.fillText('Inner ref 0.72 AU',lx,ly);
[lx,ly]=t(0.72,0.72);ctx.fillStyle='rgba(120,170,255,.75)';ctx.fillText('Mid ref 1.0 AU',lx,ly);
[lx,ly]=t(-1.05,1.1);ctx.fillStyle='rgba(255,120,90,.7)';ctx.fillText('Outer ref 1.52 AU',lx,ly)}}

// FTOP (Focal Tensor Orientation Point)
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=25;
ctx.beginPath();ctx.arc(ox,oy,Math.max(3,6*Math.min(camZ,2)),0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
circ(ctx,ox,oy,Math.max(5,10*Math.min(camZ,2)),'rgba(255,170,0,.3)',1);circ(ctx,ox,oy,Math.max(8,16*Math.min(camZ,2)),'rgba(255,170,0,.12)',0.5);
if(camZ>0.15&&DV.L){ctx.font='bold '+fs(30)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('\u25c7 FTOP',ox,oy-22*Math.min(camZ,2));
ctx.font=fs(22)+' sans-serif';ctx.fillStyle='rgba(255,200,0,.5)';ctx.fillText('Focal Tensor Origin',ox,oy+20*Math.min(camZ,2))}
{const sa=sphereAddr(0,0);
hPts.push({sx:ox,sy:oy,h:'<b>\u25c7 FTOP \u2014 Focal Tensor Orientation Point</b><br>'+
'<span class="tl">Role:</span> orientation anchor (0,0) &mdash; NOT a gravitating mass<br>'+
'<span class="tl">Sun at:</span> <span class="tv">('+(SUN_POS?SUN_POS[0].toFixed(3):'0')+', '+(SUN_POS?SUN_POS[1].toFixed(3):'0')+')</span> AU<br>'+
'<span class="tl">Sphere:</span> <span class="tv" style="color:#ffd700">'+sa.addr+'</span><br>'+
'<span class="tl">Cycle:</span> <span class="tv">'+sa.time+'</span><br>'+
'<span class="tl">Rs:</span> <span class="tv">'+Rs.toFixed(3)+' AU (&times;'+(Rs/RS_BASE).toFixed(2)+' base)</span>'})}

// Flower-of-Life — thicker, higher-contrast strokes for legibility
circ(ctx,ox,oy,Rs*.58*sc,'rgba(140,170,230,.85)',3.0);
circ(ctx,ox,oy,Rs*sc,'rgba(255,215,140,.9)',3.5);
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;const[gx,gy]=t(Rs*.58*Math.cos(a),Rs*.58*Math.sin(a));circ(ctx,gx,gy,Rs*.58*sc,'rgba(130,160,230,.75)',2.0,[5,5])}
for(let i=0;i<12;i++){const a=i*Math.PI/6;const[gx,gy]=t(Rs*Math.cos(a),Rs*Math.sin(a));circ(ctx,gx,gy,Rs*sc,'rgba(240,210,140,.55)',2.2)}
for(let i=0;i<12;i++){const a=(i*30+15)*Math.PI/180;const[gx,gy]=t(Rs*1.5*Math.cos(a),Rs*1.5*Math.sin(a));circ(ctx,gx,gy,Rs*sc,'rgba(120,160,210,.45)',1.6)}

// Cross-section pattern numbering (Fibonacci spiral overlay) — LOD based on zoom
if(DV.X){const fibPts=[];
ctx.textAlign='center';ctx.textBaseline='middle';
const showAllLabels=camZ>0.3;
const showFibLabels=camZ>0.08;
for(let k=0;k<XS.length;k++){
const[px,py]=XS[k];const[sx,sy]=t(px,py);const isFib=FIB_SET.has(k);
if(sx<-50||sx>w+50||sy<-50||sy>h+50)continue;
const xsSph=au2sph(px,py);
if(isFib){
ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=12;
const br=k>5?11:8;
ctx.beginPath();ctx.arc(sx,sy,br,0,Math.PI*2);
ctx.strokeStyle='rgba(255,170,60,.7)';ctx.lineWidth=2.5;ctx.stroke();ctx.restore();
ctx.beginPath();ctx.arc(sx,sy,3,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();
if(showFibLabels&&DV.L){ctx.font='bold '+fs(22)+' monospace';ctx.fillStyle='#ffdd44';
ctx.fillText(k,sx,sy-(br+4));
if(DV.Sph){ctx.font=fs(11)+' monospace';ctx.fillStyle='rgba(255,215,0,.5)';
ctx.fillText(xsSph.addr,sx,sy+(br+Math.round(10*FS)))}}
fibPts.push([sx,sy])}
else if(showAllLabels){ctx.beginPath();ctx.arc(sx,sy,2.5,0,Math.PI*2);ctx.fillStyle='rgba(200,175,130,.4)';ctx.fill();
if(DV.L){ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(200,175,130,.5)';ctx.fillText(k,sx,sy-Math.round(10*FS));
if(DV.Sph){ctx.font=fs(9)+' monospace';ctx.fillStyle='rgba(200,175,130,.35)';
ctx.fillText(xsSph.addr,sx,sy+Math.round(8*FS))}}}
const rr=Math.sqrt(px*px+py*py);
hPts.push({sx:sx,sy:sy,h:'<b>Cross-section #'+k+'</b><br><span class="tl">Position:</span> <span class="tv">('+px.toFixed(3)+', '+py.toFixed(3)+')</span><br><span class="tl">R(FTOP):</span> <span class="tv">'+rr.toFixed(3)+' AU</span><br><span class="tl">Sphere:</span> <span class="tv" style="color:#ffd700">'+xsSph.addr+'</span><br><span class="tl">Circles through:</span> <span class="tv">'+XS[k][2]+'</span><br><span class="tl">Fibonacci index:</span> <span class="tv">'+(isFib?'YES \u2714 (F='+k+')':'no')+'</span><br><span class="tl">Pattern:</span> <span class="tv">'+(k===0?'Origin (0/1)':k%12===0?'Full cycle (mod 12=0)':k%6===0?'Half cycle (mod 6=0)':'Spiral position '+k)+'</span>'})}
if(fibPts.length>1){ctx.beginPath();ctx.strokeStyle='rgba(255,170,60,.3)';ctx.lineWidth=1.5;ctx.setLineDash([5,5]);
ctx.moveTo(fibPts[0][0],fibPts[0][1]);
for(let i=1;i<fibPts.length;i++){
const[x0,y0]=fibPts[i-1],[x1,y1]=fibPts[i];
const mx=(x0+x1)/2+0.15*(y1-y0),my=(y0+y1)/2-0.15*(x1-x0);
ctx.quadraticCurveTo(mx,my,x1,y1)}
ctx.stroke();ctx.setLineDash([])}}

// Clock numbers on scope ring
if(DV.L){ctx.font='bold '+fs(28)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
for(let i=1;i<=12;i++){
const a=(90-i*30)*Math.PI/180;
const[cx2,cy2]=t((Rs+0.18)*Math.cos(a),(Rs+0.18)*Math.sin(a));
ctx.fillStyle=i===7?'#66aaff':'#a09070';ctx.fillText(i,cx2,cy2)}}

// Clock-face gate dots on scope ring
if(DV.G) for(let k=0;k<GXY.length;k++){
const[gx,gy]=t(GXY[k][0],GXY[k][1]);
ctx.beginPath();ctx.arc(gx,gy,5,0,Math.PI*2);ctx.fillStyle='rgba(220,185,120,.6)';ctx.fill();ctx.strokeStyle='rgba(220,185,120,.3)';ctx.lineWidth=1;ctx.stroke();
const hr=k+1;const secA1=(90-(hr-1)*30)*Math.PI/180,secA2=(90-hr*30)*Math.PI/180;
const midA=(secA1+secA2)/2;const midR=Rs*0.8;
const gPot=MU/(midR*midR);
hPts.push({sx:gx,sy:gy,h:'<b>Sector '+hr+' ('+hr+" o'clock)</b><br><span class=\"tl\">Scope pos:</span> <span class=\"tv\">("+GXY[k][0].toFixed(3)+', '+GXY[k][1].toFixed(3)+')</span><br><span class="tl">Grav potential:</span> <span class="tv">'+gPot.toFixed(1)+' AU/yr\u00b2</span><br><span class="tl">Sector angle:</span> <span class="tv">'+(90-(hr-1)*30)+'\u00b0 to '+(90-hr*30)+'\u00b0</span>'})}

// Projected tangent spheres at trajectory gates
if(DV.G) for(let k=0;k<GT.length;k++){
const[gx,gy]=t(GT[k][0],GT[k][1]);
const al=0.15+0.08*(k/11);
circ(ctx,gx,gy,0.18*sc,'rgba(220,185,120,'+al+')',1.5);
circ(ctx,gx,gy,0.09*sc,'rgba(220,185,120,'+(al*.7)+')',1)}

// Tensor mapping lines
ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(255,215,140,.55)';ctx.lineWidth=1.3;
for(let k=0;k<Math.min(GXY.length,GT.length);k++){
const[sx,sy]=t(GXY[k][0],GXY[k][1]);const[tx,ty]=t(GT[k][0],GT[k][1]);
ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(tx,ty);ctx.stroke()}
ctx.setLineDash([]);

// Trajectory — color-coded segments by speed (thicker core + outer glow for legibility)
if(DV.T){const allSpd=GI.map(g=>g.spd);const mnS=Math.min(...allSpd),mxS=Math.max(...allSpd),rngS=mxS-mnS+.001;
const spdCol=(s)=>{const t2=(s-mnS)/rngS;return'rgb('+Math.round(80+t2*175)+','+Math.round(255-t2*80)+','+Math.round(200-t2*180)+')'};
const nPerSeg=Math.floor(NT.length/13);
// Outer glow pass (wide, faint)
ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.35)';ctx.lineWidth=10;ctx.lineCap='round';
for(let i=0;i<NT.length;i++){const[sx,sy]=t(NT[i][0],NT[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}ctx.stroke();ctx.lineCap='butt';
for(let seg=0;seg<13;seg++){
const colIdx=Math.min(seg,11);const col2=seg<12?spdCol(GI[colIdx].spd):'#44ff88';
ctx.beginPath();ctx.strokeStyle=col2;ctx.lineWidth=4.5;ctx.shadowColor=col2;ctx.shadowBlur=10;
const i0=seg*nPerSeg,i1=Math.min((seg+1)*nPerSeg+1,NT.length);
for(let i=i0;i<i1;i++){const[sx,sy]=t(NT[i][0],NT[i][1]);if(i===i0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}ctx.shadowBlur=0;
// Midpoint annotations between gates (curvature + speed change)
if(DV.L){ctx.font=fs(16)+' sans-serif';ctx.textAlign='center';
for(let k=0;k<GT.length;k++){
const g=GI[k];const prev=k===0?B:GT[k-1];
const mx2=(prev[0]+GT[k][0])/2,my2=(prev[1]+GT[k][1])/2;
const[mpx,mpy]=t(mx2,my2);
ctx.fillStyle='rgba(220,200,140,.65)';
ctx.fillText('\u2220'+Math.abs(g.dhdg).toFixed(1)+'\u00b0',mpx,mpy-14);
const dsSign=g.dspd>=0?'+':'';
ctx.fillStyle=g.dspd>=0?'rgba(100,255,150,.6)':'rgba(255,130,100,.6)';
ctx.fillText(dsSign+g.dspd.toFixed(2)+' AU/yr',mpx,mpy+12)
}}
// Velocity direction arrows at each gate
for(let k=0;k<GT.length;k++){
const g=GI[k];const[gx,gy]=t(GT[k][0],GT[k][1]);
const ang=Math.atan2(g.vy,g.vx);const len=14;
const ex=gx+len*Math.cos(ang),ey=gy-len*Math.sin(ang);
ctx.beginPath();ctx.moveTo(gx,gy);ctx.lineTo(ex,ey);ctx.strokeStyle='rgba(255,210,60,.7)';ctx.lineWidth=2;ctx.stroke();
const ah=5;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-ah*Math.cos(ang-.4),ey+ah*Math.sin(ang-.4));ctx.lineTo(ex-ah*Math.cos(ang+.4),ey+ah*Math.sin(ang+.4));ctx.closePath();ctx.fillStyle='rgba(255,210,60,.7)';ctx.fill()}}
// ═══ PARALLEL OFFSET SCOPE ═══
if(showParallel&&PAR&&DV.Par){
// Parallel trajectory (green dashed)
ctx.beginPath();ctx.strokeStyle='rgba(0,220,120,.5)';ctx.lineWidth=2;ctx.setLineDash([6,4]);
const pnt=PAR.nt;for(let i=0;i<pnt.length;i++){const[sx,sy]=t(pnt[i][0],pnt[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([]);
// Parallel barrel marker
const[pbx,pby]=t(PAR.bp[0],PAR.bp[1]);
ctx.fillStyle='rgba(0,220,120,.6)';ctx.fillRect(pbx-6,pby-6,12,12);ctx.strokeStyle='#00dd88';ctx.lineWidth=1;ctx.strokeRect(pbx-6,pby-6,12,12);
if(DV.L){ctx.font='bold '+fs(20)+' sans-serif';ctx.fillStyle='#00dd88';ctx.textAlign='center';ctx.fillText('PAR',pbx,pby+24)}
// Diff arrows: primary gate → parallel gate (shows offset difference)
for(let k=0;k<PAR.gt.length;k++){
const[px,py]=t(GT[k][0],GT[k][1]);const[qx,qy]=t(PAR.gt[k][0],PAR.gt[k][1]);
// Diff arrow
ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(qx,qy);ctx.strokeStyle='rgba(0,255,140,.4)';ctx.lineWidth=1.5;ctx.stroke();
// Arrowhead
const ang2=Math.atan2(qy-py,qx-px);const ah2=5;
ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(qx-ah2*Math.cos(ang2-.35),qy-ah2*Math.sin(ang2-.35));
ctx.lineTo(qx-ah2*Math.cos(ang2+.35),qy-ah2*Math.sin(ang2+.35));ctx.closePath();ctx.fillStyle='rgba(0,255,140,.5)';ctx.fill();
// Diff label
const d=PAR.diff[k];const mx2=(px+qx)/2,my2=(py+qy)/2;
ctx.font=fs(22)+' monospace';ctx.fillStyle='rgba(0,255,160,.7)';ctx.textAlign='center';
ctx.fillText('\u0394'+d.d_pos.toFixed(4)+' AU',mx2,my2-10);
// Parallel gate dot
ctx.beginPath();ctx.arc(qx,qy,4,0,Math.PI*2);ctx.fillStyle='rgba(0,220,120,.7)';ctx.fill();
hPts.push({sx:qx,sy:qy,h:'<b>Parallel Gate '+(k+1)+' (offset +'+PAR.off+'\u00b0)</b><br><span class="tl">\u0394 Position:</span> <span class="tv">('+d.dx+', '+d.dy+')</span><br><span class="tl">\u0394 |pos|:</span> <span class="tv">'+d.d_pos+' AU</span><br><span class="tl">\u0394 Velocity:</span> <span class="tv">('+d.dvx+', '+d.dvy+')</span><br><span class="tl">\u0394 |vel|:</span> <span class="tv">'+d.d_vel+' AU/yr</span><br><span class="tl">\u0394 Heading:</span> <span class="tv">'+d.d_hdg+'\u00b0</span><br><span class="tl">Correction:</span> <span class="tv">Use \u0394v to null this offset</span>'})}
// Parallel scope label
if(DV.L){ctx.font='bold '+fs(20)+' monospace';ctx.fillStyle='#00dd88';ctx.textAlign='left';
ctx.fillText('PARALLEL SCOPE (+'+PAR.off+'\u00b0) | \u0394miss='+PAR.miss.toFixed(6)+' AU',14,h-100)}}

// ═══ MULTI-BARREL SWARM ═══
if(showSwarm&&SWM){
for(let si=0;si<SWM.length;si++){const sw=SWM[si];const col=SWM_COLS[si%SWM_COLS.length];
// Swarm trajectory
ctx.beginPath();ctx.strokeStyle=col+'88';ctx.lineWidth=1.5;ctx.setLineDash([4,6]);
const snt=sw.nt;for(let i=0;i<snt.length;i++){const[sx,sy]=t(snt[i][0],snt[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([]);
// Swarm barrel marker
const[sbx,sby]=t(sw.bp[0],sw.bp[1]);
ctx.save();ctx.beginPath();ctx.arc(sbx,sby,6,0,Math.PI*2);ctx.fillStyle=col;ctx.fill();ctx.restore();
ctx.strokeStyle=col;ctx.lineWidth=1;ctx.beginPath();ctx.arc(sbx,sby,6,0,Math.PI*2);ctx.stroke();
const svm=Math.sqrt(sw.v0[0]**2+sw.v0[1]**2);
if(DV.L){ctx.font='bold '+fs(20)+' sans-serif';ctx.fillStyle=col;ctx.textAlign='center';
ctx.fillText('B@'+sw.ck+'h',sbx,sby-18);
ctx.font=fs(16)+' monospace';ctx.fillText('|v|='+svm.toFixed(2),sbx,sby+22);
ctx.fillText('miss='+sw.miss.toExponential(1),sbx,sby+40)}
// Convergence line: barrel → target (same AU space)
const[stx,sty]=t(TG[0],TG[1]);
ctx.beginPath();ctx.setLineDash([3,8]);ctx.strokeStyle=col+'44';ctx.lineWidth=1;
ctx.moveTo(sbx,sby);ctx.lineTo(stx,sty);ctx.stroke();ctx.setLineDash([]);
// Swarm gate dots (with number labels)
for(let k2=0;k2<sw.gt.length;k2++){const[sgx,sgy]=t(sw.gt[k2][0],sw.gt[k2][1]);
ctx.beginPath();ctx.arc(sgx,sgy,3,0,Math.PI*2);ctx.fillStyle=col+'aa';ctx.fill();
if(k2%3===0&&DV.L){ctx.font=fs(14)+' sans-serif';ctx.fillStyle=col+'aa';ctx.textAlign='center';ctx.fillText(k2+1,sgx,sgy-10)}}
hPts.push({sx:sbx,sy:sby,h:'<b>Swarm Barrel @'+sw.ck+"h</b><br><span class=\"tl\">Position:</span> <span class=\"tv\">("+sw.bp[0].toFixed(4)+', '+sw.bp[1].toFixed(4)+') AU</span><br><span class="tl">|v\u2080|:</span> <span class="tv">'+svm.toFixed(4)+' AU/yr</span><br><span class="tl">Miss:</span> <span class="tv">'+sw.miss+' AU</span><br><span class="tl">v\u2080:</span> <span class="tv">('+sw.v0[0].toFixed(4)+', '+sw.v0[1].toFixed(4)+')</span><br><span class="tl">Heading:</span> <span class="tv">'+(Math.atan2(sw.v0[1],sw.v0[0])*180/Math.PI).toFixed(1)+'\u00b0</span><br><span class="tl">Role:</span> <span class="tv">Swarm convergence to target X</span>'})}
// Swarm status label
if(DV.L){ctx.font='bold '+fs(20)+' monospace';ctx.fillStyle='#cc44ff';ctx.textAlign='left';
ctx.fillText('SWARM ('+SWM.length+' barrels) | All \u2192 TARGET X | P=toggle par | S=toggle swm',14,h-120)}}

// Gate dots on trajectory with full segment detail
for(let k=0;k<GT.length;k++){
const[gx,gy]=t(GT[k][0],GT[k][1]);
ctx.beginPath();ctx.arc(gx,gy,16,0,Math.PI*2);ctx.fillStyle='#ffd700';ctx.fill();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
ctx.fillStyle='#1a1610';ctx.font='bold '+fs(20)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
ctx.fillText(k+1,gx,gy);
const g=GI[k];const prevN=k===0?'Barrel':'Gate '+(k);
hPts.push({sx:gx,sy:gy,h:'<b>Gate '+g.n+' \u2014 Correction Point</b><br><span class="tl">Position:</span> <span class="tv">('+g.pos[0]+', '+g.pos[1]+') AU</span><br><span class="tl">Speed:</span> <span class="tv">'+g.spd+' AU/yr</span><br><span class="tl">\u0394 Speed (from '+prevN+'):</span> <span class="tv" style="color:'+(g.dspd>=0?'#88ff88':'#ff8866')+'">'+( g.dspd>=0?'+':'')+g.dspd+' AU/yr</span><br><span class="tl">Heading:</span> <span class="tv">'+g.hdg+'\u00b0</span><br><span class="tl">\u0394 Heading (curve):</span> <span class="tv">'+(g.dhdg>=0?'+':'')+g.dhdg+'\u00b0</span><br><span class="tl">Curvature:</span> <span class="tv">'+g.curv+' /AU</span><br><span class="tl">Arc from '+prevN+':</span> <span class="tv">'+g.arc+' AU</span><br><span class="tl">R(focal):</span> <span class="tv">'+g.rsun+' AU</span><br><span class="tl">Gravity:</span> <span class="tv">'+g.grav+' AU/yr\u00b2</span><br><span class="tl">Time:</span> <span class="tv">+'+g.tg+' yr ('+g.tr+' rem)</span><br><span class="tl">Jacobian cond:</span> <span class="tv">'+g.jc+'</span>'})}

// Barrel
const[bx,by]=t(B[0],B[1]);
if(DV.Bar){
ctx.fillStyle='#4488ff';ctx.fillRect(bx-9,by-9,18,18);ctx.strokeStyle='#88bbff';ctx.lineWidth=1.5;ctx.strokeRect(bx-9,by-9,18,18);
if(DV.L){ctx.font='bold '+fs(22)+' sans-serif';ctx.fillStyle='#4488ff';ctx.textAlign='center';
ctx.fillText("BARREL (7 o'clock)",bx,by+34)}}

// Solar system — sun + 20 planets + moons (perturbing masses).
// Stored on the canvas so the click/hover handlers can hit-test against them.
c._solarHits=drawSolarSystem2D(ctx,t,{scale:sc});
// Feed every solar body into hPts with a sphere-nav address so hover works.
for(const bh of (c._solarHits||[])){const sa=sphereAddr(bh.body.x,bh.body.y);
 const isSun=bh.body.kind==='sun',isMoon=bh.body.kind==='moon';
 hPts.push({sx:bh.x,sy:bh.y,h:'<b style="color:'+bh.body.color+'">'+(isSun?'☼ ':(isMoon?'• ':'● '))+bh.body.name+'</b><br>'+
  '<span class="tl">Kind:</span> <span class="tv">'+bh.body.kind+'</span><br>'+
  '<span class="tl">Position:</span> <span class="tv">('+bh.body.x.toFixed(4)+', '+bh.body.y.toFixed(4)+')</span> AU<br>'+
  '<span class="tl">Sphere:</span> <span class="tv" style="color:#ffd700">'+sa.addr+'</span><br>'+
  (isSun?'':'<span class="tl">Orbit r:</span> <span class="tv">'+bh.body.r_orbit.toFixed(4)+'</span> AU · <span class="tl">T:</span> <span class="tv">'+bh.body.period.toFixed(3)+'</span> yr<br>')+
  '<span class="tl">&mu;:</span> <span class="tv">'+bh.body.mu.toExponential(3)+'</span> AU³/yr² ('+(bh.body.mu/MU*100).toExponential(2)+'% of Sun)<br>'+
  '<span class="tl">Click:</span> <span class="tv">set as TARGET</span>'})}
// Target — pans with scope (same AU coordinate space)
if(DV.Tgt){const[tgx,tgy]=t(TG[0],TG[1]);
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=15;
ctx.beginPath();ctx.arc(tgx,tgy,13,0,Math.PI*2);ctx.fillStyle='#44ff88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tgx,tgy,13,0,Math.PI*2);ctx.stroke();
circ(ctx,tgx,tgy,HR*sc,'rgba(68,255,136,.3)',1,[4,4]);
// Crosshair on target
ctx.strokeStyle='rgba(68,255,136,.4)';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(tgx-18,tgy);ctx.lineTo(tgx+18,tgy);ctx.stroke();
ctx.beginPath();ctx.moveTo(tgx,tgy-18);ctx.lineTo(tgx,tgy+18);ctx.stroke();
// Gap line barrel → target
const gapDist=Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2);
ctx.beginPath();ctx.setLineDash([6,4]);ctx.strokeStyle='rgba(68,255,136,.3)';ctx.lineWidth=1;
ctx.moveTo(bx,by);ctx.lineTo(tgx,tgy);ctx.stroke();ctx.setLineDash([]);
const gmx=(bx+tgx)/2,gmy=(by+tgy)/2;
ctx.font='bold '+fs(28)+' monospace';ctx.fillStyle='rgba(68,255,136,.6)';ctx.textAlign='center';
ctx.fillText(gapDist.toFixed(3)+' AU',gmx,gmy-12);
// Target label with AU coords
const tgR=Math.sqrt(TG[0]*TG[0]+TG[1]*TG[1]);
if(DV.L){ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='left';
ctx.fillText('TARGET',tgx+24,tgy-10);
ctx.font=fs(18)+' monospace';ctx.fillText('('+TG[0].toFixed(2)+', '+TG[1].toFixed(2)+') AU  r='+tgR.toFixed(3),tgx+24,tgy+18);
ctx.fillText('Gap: '+gapDist.toFixed(4)+' AU from barrel',tgx+24,tgy+44);
if(TGT_ORB){ctx.fillStyle='#88ccaa';ctx.fillText('v_orb='+TGT_ORB.spd+' AU/yr ('+TGT_ORB.spd_kms+' km/s) P='+TGT_ORB.period+' yr',tgx+24,tgy+66);
ctx.fillStyle='#886';ctx.fillText('\u03b2='+TGT_ORB.beta.toExponential(2)+' \u03b3='+TGT_ORB.gamma.toFixed(10),tgx+24,tgy+84)}}
// Target orbital history path (dashed orange/red arc — where target came from)
if(TGT_ORB&&TGT_ORB.history){
ctx.beginPath();ctx.strokeStyle='rgba(255,140,60,.22)';ctx.lineWidth=1.5;ctx.setLineDash([4,6]);
const stepH=Math.max(1,Math.floor(TGT_ORB.history.length/300));
for(let ih=0;ih<TGT_ORB.history.length;ih+=stepH){const[hpx,hpy]=t(TGT_ORB.history[ih][0],TGT_ORB.history[ih][1]);
if(ih===0)ctx.moveTo(hpx,hpy);else ctx.lineTo(hpx,hpy)}
ctx.stroke();ctx.setLineDash([])}
// Target orbital predicted path (dashed green arc — where target is going)
if(TGT_ORB&&TGT_ORB.path){
ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.25)';ctx.lineWidth=1.5;ctx.setLineDash([6,6]);
const step2=Math.max(1,Math.floor(TGT_ORB.path.length/300));
for(let i2=0;i2<TGT_ORB.path.length;i2+=step2){const[px,py]=t(TGT_ORB.path[i2][0],TGT_ORB.path[i2][1]);
if(i2===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
ctx.stroke();ctx.setLineDash([]);
// Target at T_f marker (where target will be at intercept time)
const[tfx,tfy]=t(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=8;
ctx.beginPath();ctx.arc(tfx,tfy,8,0,Math.PI*2);ctx.strokeStyle='#44ff88';ctx.lineWidth=2;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);ctx.restore();
if(DV.L){ctx.font=fs(16)+' monospace';ctx.fillStyle='rgba(68,255,136,.6)';ctx.textAlign='center';
ctx.fillText('T@Tf('+TGT_ORB.at_tf[0].toFixed(2)+','+TGT_ORB.at_tf[1].toFixed(2)+')',tfx,tfy+22)}
// Target gate-time positions (where target IS at each gate)
for(let kg=0;kg<TGT_ORB.gates.length;kg++){const[gsx,gsy]=t(TGT_ORB.gates[kg][0],TGT_ORB.gates[kg][1]);
ctx.beginPath();ctx.arc(gsx,gsy,3,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.3)';ctx.fill()}}
// Target motion label
if(DV.L&&DV.En){ctx.font=fs(16)+' monospace';ctx.fillStyle='rgba(68,255,136,.5)';ctx.textAlign='left';
ctx.fillText('v_orb='+TGT_ORB.spd+' AU/yr ('+TGT_ORB.spd_kms+' km/s) P='+TGT_ORB.period+' yr',tgx+24,tgy+68)}
hPts.push({sx:tgx,sy:tgy,h:'<b>TARGET \u2014 Location X</b><br><span class="tl">Position:</span> <span class="tv">('+TG[0].toFixed(4)+', '+TG[1].toFixed(4)+') AU</span><br><span class="tl">R(FTOP):</span> <span class="tv">'+tgR.toFixed(4)+' AU</span><br><span class="tl">Zone:</span> <span class="tv">Outer reference orbit</span><br><span class="tl">Hit radius:</span> <span class="tv">'+HR+' AU ('+(HR*1.496e5).toFixed(0)+' km)</span><br><span class="tl">Gap from barrel:</span> <span class="tv">'+gapDist.toFixed(4)+' AU</span><br><span class="tl">Heading(B\u2192T):</span> <span class="tv">'+(Math.atan2(TG[1]-B[1],TG[0]-B[0])*180/Math.PI).toFixed(1)+'\u00b0</span><br><span class="tl">Orbital v:</span> <span class="tv">'+TGT_ORB.spd+' AU/yr ('+TGT_ORB.spd_kms+' km/s)</span><br><span class="tl">Period:</span> <span class="tv">'+TGT_ORB.period+' yr</span><br><span class="tl">At T_f:</span> <span class="tv">('+TGT_ORB.at_tf[0].toFixed(4)+', '+TGT_ORB.at_tf[1].toFixed(4)+')</span>'})
} // end DV.Tgt

// Animated comet marker
if(animT>0){const ap=getAnimPos();const[ax,ay]=t(ap[0],ap[1]);
// Trail (previous 20 positions)
const f0=animT/TF;const nPts=NT.length-1;
for(let tt=1;tt<=15;tt++){const ft=Math.max(0,f0-tt*0.003);const ii=ft*nPts;const i0=Math.floor(ii),i1=Math.min(i0+1,nPts);const frc=ii-i0;
const tx2=NT[i0][0]+(NT[i1][0]-NT[i0][0])*frc,ty2=NT[i0][1]+(NT[i1][1]-NT[i0][1])*frc;
const[px2,py2]=t(tx2,ty2);const al=0.6-tt*0.04;
ctx.beginPath();ctx.arc(px2,py2,4-tt*0.2,0,Math.PI*2);ctx.fillStyle='rgba(255,150,40,'+Math.max(0,al)+')';ctx.fill()}
// Comet head
ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=30;
ctx.beginPath();ctx.arc(ax,ay,8,0,Math.PI*2);ctx.fillStyle='#ffaa00';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(ax,ay,8,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
ctx.font='bold '+fs(26)+' monospace';ctx.fillStyle='#fff';ctx.textAlign='left';
ctx.fillText('T+'+animT.toFixed(3)+'yr',ax+18,ay-8);ctx.fillStyle='#ffaa00';
ctx.fillText(getAnimSpd().toFixed(2)+' AU/yr',ax+18,ay+20)}

// AU scale ruler — adaptive to zoom level
{const targetPx=120;const rawAU=targetPx/sc;
const e2=Math.pow(10,Math.floor(Math.log10(rawAU)));let niceAU=e2;
if(rawAU/e2>=5)niceAU=5*e2;else if(rawAU/e2>=2)niceAU=2*e2;
const rpx=niceAU*sc;
ctx.strokeStyle='#c8a96e';ctx.lineWidth=2;
ctx.beginPath();ctx.moveTo(20,h-58);ctx.lineTo(20+rpx,h-58);ctx.stroke();
ctx.beginPath();ctx.moveTo(20,h-62);ctx.lineTo(20,h-54);ctx.stroke();
ctx.beginPath();ctx.moveTo(20+rpx,h-62);ctx.lineTo(20+rpx,h-54);ctx.stroke();
ctx.font='bold '+fs(20)+' monospace';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
const[unitLbl]=scaleUnit(niceAU);
ctx.fillText(niceAU<1e4?niceAU.toFixed(niceAU<0.1?3:niceAU<1?2:1)+' AU':unitLbl,20+rpx/2,h-72)
ctx.font=fs(16)+' monospace';ctx.fillStyle='#889';ctx.textAlign='left';
ctx.fillText('Zoom: '+camZ.toFixed(camZ<1?3:1)+'x | Cam: ('+camX.toFixed(3)+','+camY.toFixed(3)+') AU',20,h-44)}

// Title + scope-distance readout (Rs, diameter, grow factor vs base stencil).
// The scope is a SCALE UNIT \u2014 Rs grows so the whole scene fits inside ONE
// template. Showing it prominently here reminds the viewer at what scale
// they're looking at the corridor.
if(DV.L){ctx.font='bold '+fs(26)+' "Times New Roman",serif';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
ctx.fillText('2-D Tensor Scope \u2014 3D Corridor Reduced to 2D',w/2,36);
ctx.font=fs(18)+' sans-serif';ctx.fillStyle='#889';
ctx.fillText('Flower-of-Life \u00b7 12 gates \u00b7 trajectory overlay \u00b7 hover for data \u00b7 SPACE=play',w/2,62);
// Scope scale line \u2014 bright gold so it reads as a value, not chrome
const grow=Rs/Math.max(RS_BASE,1e-9);
const scopeLine='\u25cb Scope R'+'\u209b'+' = '+Rs.toFixed(3)+' AU  \u00b7  \u2300 '+(2*Rs).toFixed(3)+' AU  \u00b7  '+grow.toFixed(2)+'\u00d7 base ('+RS_BASE+' AU)  \u00b7  Zoom '+camZ.toFixed(camZ<1?3:2)+'\u00d7';
ctx.font='bold '+fs(16)+' "Consolas",monospace';ctx.fillStyle='#ffd700';
ctx.fillText(scopeLine,w/2,86)}

// Legend — top-left to avoid obstructing AU ruler
// Legend moved to shared clickable #legend overlay (bottom-left fixed panel).
ctx.textAlign='right';ctx.font='bold '+fs(20)+' monospace';
ctx.fillStyle='#44ff88';ctx.fillText('Hit:'+R.ch+'% | Miss:'+R.cmm+' AU',w-14,h-14);
ctx.fillStyle='#ff6644';ctx.fillText('Baseline:'+R.bh+'% | Miss:'+R.bmm+' AU',w-14,h-44);
ctx.fillStyle='#667';ctx.font=fs(16)+' sans-serif';ctx.fillText('RClick=pan | Scroll=zoom | ESC=info',w-14,h-74)}

// \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
// PARALLEL SCOPE tab \u2014 parallel trajectory shown at full fidelity as the
// primary of its own scope (so the user can study it the way they study
// the main scope, with its own camera and live-math panel).
// \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
let parCamX=0,parCamY=0,parCamZ=1,parPanDrg=false,parLmx=0,parLmy=0;
// SALVO tab camera: world-coord pan + zoom-toward-cursor.
let salCamX=0,salCamY=0,salCamZ=1,salPanDrg=false,salLmx=0,salLmy=0;
function drawParallel(){
const c=document.getElementById('parC');if(!c)return;
const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
ctx.fillStyle='#0a1014';ctx.fillRect(0,0,w,h);
if(!PAR){ctx.fillStyle='#c8a96e';ctx.font='bold 14px monospace';ctx.textAlign='center';
ctx.fillText('No parallel scope data available.',w/2,h/2);return}
// Normalise by Rs/Rs_base so an auto-grown scope still fills the viewport.
const baseSc=Math.min(w,h)/5.6*(RS_BASE/Math.max(Rs,1e-9));const sc=baseSc*parCamZ;
const cx=w/2,cy=h/2;
const t=(x,y)=>[cx+(x-parCamX)*sc,cy-(y-parCamY)*sc];
const PB=PAR.bp;
// Transparent 3D flower-of-life sphere grid (the shared stencil expansion)
if(DV.X){const to3d=(x,y,z)=>t(x,y);drawFoL3DSpheres(ctx,to3d,{alpha:0.18,use3D:false})}
// Solar system (sun + 20 planets + moons) — the gravitational field perturbing the parallel trajectory.
c._solarHits=drawSolarSystem2D(ctx,t,{scale:sc*0.8});
// Orbit rings
if(DV.O){const[ox,oy]=t(0,0);
circ(ctx,ox,oy,0.72*sc,'rgba(210,180,100,.35)',1.5,[6,4]);
circ(ctx,ox,oy,1.0*sc,'rgba(120,170,255,.4)',1.5,[6,4]);
circ(ctx,ox,oy,1.52*sc,'rgba(255,120,90,.35)',1.5,[6,4]);
if(DV.L){ctx.font='bold '+fs(24)+' sans-serif';ctx.textAlign='left';
let[lx,ly]=t(0.51,0.51);ctx.fillStyle='rgba(210,180,100,.7)';ctx.fillText('Inner 0.72 AU',lx,ly);
[lx,ly]=t(0.72,0.72);ctx.fillStyle='rgba(120,170,255,.7)';ctx.fillText('Mid 1.0 AU',lx,ly);
[lx,ly]=t(-1.05,1.1);ctx.fillStyle='rgba(255,120,90,.7)';ctx.fillText('Outer 1.52 AU',lx,ly)}}
// FTOP
const[ox,oy]=t(0,0);
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=24;
ctx.beginPath();ctx.arc(ox,oy,6,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
if(DV.L){ctx.font='bold '+fs(28)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('\u25c7 FTOP',ox,oy-18)}
// Flower-of-Life lattice (same structure as main scope)
circ(ctx,ox,oy,Rs*.58*sc,'rgba(100,120,170,.45)',1.6);
circ(ctx,ox,oy,Rs*sc,'rgba(220,185,120,.55)',2.4);
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;const[gx,gy]=t(Rs*.58*Math.cos(a),Rs*.58*Math.sin(a));circ(ctx,gx,gy,Rs*.58*sc,'rgba(80,100,150,.35)',1,[5,5])}
for(let i=0;i<12;i++){const a=i*Math.PI/6;const[gx,gy]=t(Rs*Math.cos(a),Rs*Math.sin(a));circ(ctx,gx,gy,Rs*sc,'rgba(220,185,120,.28)',1.2)}
// Clock numbers
if(DV.L){ctx.font='bold '+fs(26)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
for(let i=1;i<=12;i++){const a=(90-i*30)*Math.PI/180;const[cx2,cy2]=t((Rs+0.18)*Math.cos(a),(Rs+0.18)*Math.sin(a));
ctx.fillStyle=i===7?'#66aaff':'#a09070';ctx.fillText(i,cx2,cy2)}}
// Clock-face gate dots
if(DV.G)for(let k=0;k<GXY.length;k++){const[gx,gy]=t(GXY[k][0],GXY[k][1]);
ctx.beginPath();ctx.arc(gx,gy,5,0,Math.PI*2);ctx.fillStyle='rgba(220,185,120,.55)';ctx.fill()}
// Projected tangent spheres at each trajectory gate (for parallel gates, green-tinted)
if(DV.G)for(let k=0;k<PAR.gt.length;k++){const[gx,gy]=t(PAR.gt[k][0],PAR.gt[k][1]);
const al=0.18+0.08*(k/11);circ(ctx,gx,gy,0.18*sc,'rgba(0,220,140,'+al+')',1.5);
circ(ctx,gx,gy,0.09*sc,'rgba(0,220,140,'+(al*.7)+')',1)}
// Tensor mapping lines scope-gate \u2192 parallel-trajectory-gate
ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(80,255,170,.55)';ctx.lineWidth=1.4;
for(let k=0;k<Math.min(GXY.length,PAR.gt.length);k++){
const[sx,sy]=t(GXY[k][0],GXY[k][1]);const[tx,ty]=t(PAR.gt[k][0],PAR.gt[k][1]);
ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(tx,ty);ctx.stroke()}ctx.setLineDash([]);
// PRIMARY trajectory \u2014 dimmed reference (so the viewer can see both)
if(DV.T&&NT){ctx.beginPath();ctx.strokeStyle='rgba(255,215,0,.28)';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
for(let i=0;i<NT.length;i++){const[sx,sy]=t(NT[i][0],NT[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([])}
// PARALLEL trajectory \u2014 the star of this tab, speed-colored + glow
if(DV.T){const allSpd=GI.map(g=>g.spd);const mn=Math.min(...allSpd),mx=Math.max(...allSpd),rng=mx-mn+.001;
const col=(s)=>{const t2=(s-mn)/rng;return'rgb('+Math.round(40+t2*80)+','+Math.round(255-t2*40)+','+Math.round(160-t2*60)+')'};
const pnt=PAR.nt;
// Outer glow
ctx.beginPath();ctx.strokeStyle='rgba(0,255,140,.18)';ctx.lineWidth=9;ctx.lineCap='round';
for(let i=0;i<pnt.length;i++){const[sx,sy]=t(pnt[i][0],pnt[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}ctx.stroke();
// Speed-colored core
const nPerSeg=Math.floor(pnt.length/13);
for(let seg=0;seg<13;seg++){const cIdx=Math.min(seg,11);const co=seg<12?col(GI[cIdx].spd):'#44ffaa';
ctx.beginPath();ctx.strokeStyle=co;ctx.lineWidth=3.5;ctx.shadowColor=co;ctx.shadowBlur=8;
const i0=seg*nPerSeg,i1=Math.min((seg+1)*nPerSeg+1,pnt.length);
for(let i=i0;i<i1;i++){const[sx,sy]=t(pnt[i][0],pnt[i][1]);if(i===i0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}ctx.shadowBlur=0}
// Primary barrel (dim reference)
{const[bsx,bsy]=t(B[0],B[1]);ctx.fillStyle='rgba(255,215,0,.35)';ctx.fillRect(bsx-5,bsy-5,10,10);
ctx.strokeStyle='rgba(255,215,0,.5)';ctx.strokeRect(bsx-5,bsy-5,10,10);
if(DV.L){ctx.font='bold '+fs(18)+' sans-serif';ctx.fillStyle='rgba(255,215,0,.6)';ctx.textAlign='center';ctx.fillText('PRIMARY (ref)',bsx,bsy-12)}}
// PARALLEL barrel \u2014 bright
{const[pbx,pby]=t(PB[0],PB[1]);ctx.save();ctx.shadowColor='#00ff90';ctx.shadowBlur=18;
ctx.fillStyle='#00dd88';ctx.fillRect(pbx-8,pby-8,16,16);ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(pbx-8,pby-8,16,16);
if(DV.L){ctx.font='bold '+fs(26)+' sans-serif';ctx.fillStyle='#00ff90';ctx.textAlign='center';ctx.fillText('PARALLEL (+'+PAR.off+'\u00b0)',pbx,pby+28);
ctx.font=fs(20)+' monospace';ctx.fillText('('+PB[0].toFixed(3)+', '+PB[1].toFixed(3)+')',pbx,pby+46)}}
// PARALLEL gate markers with \u0394 labels
for(let k=0;k<PAR.gt.length;k++){const[gx,gy]=t(PAR.gt[k][0],PAR.gt[k][1]);const d=PAR.diff[k];
// bright gate dot
ctx.save();ctx.shadowColor='#00ff90';ctx.shadowBlur=10;
ctx.beginPath();ctx.arc(gx,gy,6,0,Math.PI*2);ctx.fillStyle='#00dd88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke();
ctx.fillStyle='#041810';ctx.font='bold '+fs(18)+' monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(k+1,gx,gy);
ctx.textBaseline='alphabetic';
// velocity arrow
const g=GI[k];const[qx,qy]=t(GT[k][0],GT[k][1]);
ctx.beginPath();ctx.moveTo(qx,qy);ctx.lineTo(gx,gy);ctx.strokeStyle='rgba(0,255,140,.5)';ctx.lineWidth=1.2;ctx.setLineDash([2,3]);ctx.stroke();ctx.setLineDash([]);
// \u0394 text
if(DV.L){ctx.font=fs(16)+' monospace';ctx.fillStyle='rgba(0,255,160,.85)';ctx.textAlign='left';
ctx.fillText('\u0394'+d.d_pos.toFixed(4)+' AU',gx+10,gy-4);
ctx.fillStyle='rgba(170,220,255,.7)';
ctx.fillText('\u0394v '+d.d_vel.toFixed(4)+' AU/yr',gx+10,gy+12);
ctx.fillStyle='rgba(255,210,120,.7)';
ctx.fillText('\u0394hdg '+d.d_hdg.toFixed(2)+'\u00b0',gx+10,gy+26)}}
// Primary gate dots (dim)
for(let k=0;k<GT.length;k++){const[qx,qy]=t(GT[k][0],GT[k][1]);
ctx.beginPath();ctx.arc(qx,qy,3,0,Math.PI*2);ctx.fillStyle='rgba(255,215,0,.4)';ctx.fill()}
// Target
{const[tgx,tgy]=t(TG[0],TG[1]);ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=20;
ctx.beginPath();ctx.arc(tgx,tgy,14,0,Math.PI*2);ctx.fillStyle='#44ff88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
if(DV.L){ctx.font='bold '+fs(26)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';ctx.fillText('TARGET X',tgx,tgy-26)}}
// Animated parallel comet at current animT
{const f=animT/TF;const idx=f*(PAR.nt.length-1);
const i0=Math.floor(idx),i1=Math.min(i0+1,PAR.nt.length-1);const tt=idx-i0;
const pxP=PAR.nt[i0][0]+(PAR.nt[i1][0]-PAR.nt[i0][0])*tt;
const pyP=PAR.nt[i0][1]+(PAR.nt[i1][1]-PAR.nt[i0][1])*tt;
const[sx,sy]=t(pxP,pyP);
ctx.save();ctx.shadowColor='#aaffdd';ctx.shadowBlur=14;
ctx.beginPath();ctx.arc(sx,sy,6,0,Math.PI*2);ctx.fillStyle='#ccffee';ctx.fill();ctx.restore()}
// Header
ctx.textAlign='center';ctx.font='bold '+fs(26)+' serif';ctx.fillStyle='#00dd88';
ctx.fillText('PARALLEL SCOPE  \u00b7  offset +'+PAR.off+'\u00b0  \u00b7  \u0394miss '+PAR.miss.toFixed(6)+' AU',w/2,30);
ctx.font=fs(16)+' monospace';ctx.fillStyle='#88c8a8';
ctx.fillText('Directional derivative of the transfer solution wrt barrel angle \u2014 use \u0394 at each gate as a pre-computed correction.',w/2,54);
// Hover hit data: gates + barrels + target + trajectory samples
{const hits=[];
for(let k=0;k<PAR.gt.length;k++){const[gx,gy]=t(PAR.gt[k][0],PAR.gt[k][1]);
hits.push({sx:gx,sy:gy,r:12,type:'pgate',k:k})}
{const[bsx,bsy]=t(PB[0],PB[1]);hits.push({sx:bsx,sy:bsy,r:14,type:'pbar'})}
{const[bsx,bsy]=t(B[0],B[1]);hits.push({sx:bsx,sy:bsy,r:10,type:'primbar'})}
{const[tgx,tgy]=t(TG[0],TG[1]);hits.push({sx:tgx,sy:tgy,r:16,type:'tgt'})}
const step=Math.max(1,Math.floor(PAR.nt.length/160));
const trj=[];for(let i=0;i<PAR.nt.length;i+=step){const[sx,sy]=t(PAR.nt[i][0],PAR.nt[i][1]);
trj.push({sx:sx,sy:sy,wx:PAR.nt[i][0],wy:PAR.nt[i][1],ti:i/(PAR.nt.length-1)*TF})}
c._parHits=hits;c._parTraj=trj}
updateParHud()}

// Separate hud so both tabs can share the expanded math dossier.
function updateParHud(){const hel=document.getElementById('parHud');if(!hel||!PAR)return;
const gnow=getAnimGate();const gk=Math.min(gnow-1,11);
// Animated parallel position + velocity (use nearest gate state for v)
const f=animT/TF;const idx=f*(PAR.nt.length-1);
const i0=Math.floor(idx),i1=Math.min(i0+1,PAR.nt.length-1);const tt=idx-i0;
const pxP=PAR.nt[i0][0]+(PAR.nt[i1][0]-PAR.nt[i0][0])*tt;
const pyP=PAR.nt[i0][1]+(PAR.nt[i1][1]-PAR.nt[i0][1])*tt;
// parallel velocity: approximate from adjacent samples
const dt=TF/Math.max(1,PAR.nt.length-1);
const vxP=(PAR.nt[Math.min(i0+1,PAR.nt.length-1)][0]-PAR.nt[Math.max(i0-1,0)][0])/(2*dt);
const vyP=(PAR.nt[Math.min(i0+1,PAR.nt.length-1)][1]-PAR.nt[Math.max(i0-1,0)][1])/(2*dt);
const km=liveKinematics([pxP,pyP],[vxP,vyP],animT);
const pm=liveKinematics(getAnimPos(),[GI[gk].vx,GI[gk].vy],animT);
const d=PAR.diff[gk];
let h='<b style="color:#00dd88">LIVE PARALLEL TRAJECTORY</b><br>';
h+='<span class="hd">T:</span> <span class="hl" style="color:#aaffcc">+'+animT.toFixed(4)+' yr</span> ('+((animT/TF)*100).toFixed(1)+'%)<br>';
h+='<span class="hd">Pos:</span> <span class="hl" style="color:#aaffcc">('+pxP.toFixed(4)+', '+pyP.toFixed(4)+')</span> AU<br>';
h+='<span class="hd">Speed:</span> <span class="hl" style="color:#aaffcc">'+km.spd.toFixed(3)+'</span> AU/yr ('+km.spd_kms.toFixed(2)+' km/s)<br>';
h+='<span class="hd">R(FTOP):</span> <span class="hl" style="color:#aaffcc">'+km.r.toFixed(4)+'</span> AU<br>';
h+='<span class="hd">Gate:</span> <span class="hl" style="color:#aaffcc">'+gnow+'/12</span><br>';
h+='<span style="color:#00cc88;font:bold 8px monospace">\u2500\u2500 ORBIT MECHANICS \u2500\u2500</span><br>';
h+='<span class="hd">\u03b5:</span> <span class="hl" style="color:#aaffcc">'+km.eps.toFixed(4)+'</span> AU\u00b2/yr\u00b2<br>';
h+='<span class="hd">L:</span> <span class="hl" style="color:#aaffcc">'+km.L.toFixed(5)+'</span> AU\u00b2/yr<br>';
h+='<span class="hd">a:</span> <span class="hl" style="color:#aaffcc">'+(isFinite(km.sma)?km.sma.toFixed(4):'\u221e')+'</span> AU  <span class="hd">e:</span> <span class="hl" style="color:#aaffcc">'+km.ecc.toFixed(4)+'</span><br>';
h+='<span class="hd">\u03b3:</span> <span class="hl" style="color:#aaffcc">'+km.gamma.toFixed(2)+'\u00b0</span>  <span class="hd">v/v<sub>esc</sub>:</span> <span class="hl" style="color:#aaffcc">'+(km.spd/km.vesc).toFixed(3)+'</span><br>';
h+='<span class="hd">|a|:</span> <span class="hl" style="color:#aaffcc">'+km.ag.toFixed(4)+'</span> AU/yr\u00b2 ('+km.ag_mms2.toFixed(3)+' mm/s\u00b2)<br>';
h+='<span class="hd">|jerk|:</span> <span class="hl" style="color:#aaffcc">'+km.jerk.toFixed(4)+'</span> AU/yr\u00b3<br>';
// \u0394 vs primary at this moment
h+='<span style="color:#00cc88;font:bold 8px monospace">\u2500\u2500 \u0394 vs PRIMARY \u2500\u2500</span><br>';
h+='<span class="hd">\u0394pos (gate):</span> <span class="hl">'+d.d_pos.toFixed(5)+' AU</span><br>';
h+='<span class="hd">\u0394vel (gate):</span> <span class="hl">'+d.d_vel.toFixed(5)+' AU/yr</span><br>';
h+='<span class="hd">\u0394hdg (gate):</span> <span class="hl">'+d.d_hdg.toFixed(2)+'\u00b0</span><br>';
h+='<span class="hd">\u0394\u03b5:</span> <span class="hl">'+(km.eps-pm.eps).toFixed(5)+' AU\u00b2/yr\u00b2</span><br>';
h+='<span class="hd">\u0394L:</span> <span class="hl">'+(km.L-pm.L).toFixed(5)+' AU\u00b2/yr</span><br>';
h+='<span class="hd">\u0394a:</span> <span class="hl">'+((isFinite(km.sma)&&isFinite(pm.sma))?(km.sma-pm.sma).toFixed(5):'\u2014')+' AU</span><br>';
h+='<span class="hd">\u0394e:</span> <span class="hl">'+(km.ecc-pm.ecc).toFixed(5)+'</span><br>';
// Full 12-gate \u0394 table
h+='<span style="color:#00cc88;font:bold 8px monospace">\u2500\u2500 12-GATE \u0394 TABLE \u2500\u2500</span>';
h+='<table style="width:100%;border-collapse:collapse;font:9px monospace;color:#c8f5d8;margin-top:4px">';
h+='<thead><tr style="color:#00dd88;border-bottom:1px solid #0a5"><th align="left">k</th><th align="right">\u0394pos</th><th align="right">\u0394vel</th><th align="right">\u0394hdg</th></tr></thead><tbody>';
for(let k=0;k<PAR.diff.length;k++){const dd=PAR.diff[k];const mark=k===gk?' style="background:rgba(0,220,140,.14)"':'';
h+='<tr'+mark+'><td>'+(k+1)+'</td><td align="right">'+dd.d_pos.toFixed(4)+'</td><td align="right">'+dd.d_vel.toFixed(4)+'</td><td align="right">'+dd.d_hdg.toFixed(2)+'\u00b0</td></tr>'}
h+='</tbody></table>';
// Parallel barrel energy/power
if(PAR.proj){const pr=PAR.proj;
h+='<span style="color:#ffd700;font:bold 8px monospace">\u2500\u2500 PROJECTILE &amp; ENERGY \u2500\u2500</span><br>';
h+='<span class="hd">\u2300:</span> <span class="hl">'+pr.diameter_m+' m</span>  <span class="hd">m:</span> <span class="hl">'+pr.mass_kg.toExponential(3)+' kg</span><br>';
h+='<span class="hd">KE:</span> <span class="hl">'+pr.ke_j.toExponential(3)+' J</span> ('+pr.tnt_kt.toFixed(2)+' kt)<br>';
h+='<span class="hd">Total E:</span> <span class="hl">'+pr.total_j.toExponential(3)+' J</span><br>';
h+='<span class="hd">P<sub>avg</sub>:</span> <span class="hl">'+pr.power_avg_W.toExponential(3)+' W</span><br>';
h+='<span class="hd">\u0394v total:</span> <span class="hl">'+pr.dv_total_kms.toFixed(2)+' km/s</span>  <span class="hd">fuel:</span> <span class="hl">'+(pr.fuel_frac*100).toFixed(2)+'%</span>'}
hel.innerHTML=h;
// Bottom coord readout
const coor=document.getElementById('parCoord');
if(coor)coor.textContent='PARALLEL comet | R='+km.r.toFixed(3)+' AU | v='+km.spd.toFixed(3)+' AU/yr | gate '+gnow+'/12 | \u0394pos '+d.d_pos.toFixed(4)+' AU'}

// \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
// PARALLEL 3D (tab 12) \u2014 draw3D-style corridor but with PARALLEL barrel /
// trajectory / gates as the primary of their own scope. Primary trajectory
// is kept as a dimmed reference so the \u0394 geometry reads at a glance.
// Independent rotation/zoom state (p3rx/p3ry/p3z).
// \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
function drawParallel3D(){
const c=document.getElementById('p3dC');if(!c)return;
const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
const cx=w/2,cy=h/2,sc=Math.min(w,h)/4.5*(RS_BASE/Math.max(Rs,1e-9));
ctx.fillStyle='#071212';ctx.fillRect(0,0,w,h);
if(!PAR){ctx.fillStyle='#c8a96e';ctx.font='bold 14px monospace';ctx.textAlign='center';
ctx.fillText('No parallel scope data available.',w/2,h/2);return}
const tsc=2.0/TF;
const pj=(x,y,z)=>projR(x,y,z,cx,cy,sc,p3rx,p3ry,p3z);
// Floor grid
if(DV.Grid){ctx.strokeStyle='rgba(140,195,165,.40)';ctx.lineWidth=0.9;
for(let g=-2;g<=2;g+=0.5){
const[a1,b1]=pj(g,-2,0);const[a2,b2]=pj(g,2,0);ctx.beginPath();ctx.moveTo(a1,b1);ctx.lineTo(a2,b2);ctx.stroke();
const[c1,d1]=pj(-2,g,0);const[c2,d2]=pj(2,g,0);ctx.beginPath();ctx.moveTo(c1,d1);ctx.lineTo(c2,d2);ctx.stroke()}}
// Axes
const[axX,axY]=pj(2.2,0,0);ctx.font='bold '+fs(22)+' monospace';ctx.fillStyle='rgba(180,240,200,.55)';ctx.textAlign='center';ctx.fillText('X (AU)',axX,axY);
const[ayX,ayY]=pj(0,2.2,0);ctx.fillText('Y (AU)',ayX,ayY);
const[azX,azY]=pj(0,0,TF*tsc+0.2);ctx.fillText('T (yr)',azX,azY);
ctx.beginPath();ctx.strokeStyle='rgba(180,240,200,.3)';ctx.lineWidth=1;ctx.setLineDash([4,6]);
const[z0x,z0y]=pj(0,0,0);const[z1x,z1y]=pj(0,0,TF*tsc);
ctx.moveTo(z0x,z0y);ctx.lineTo(z1x,z1y);ctx.stroke();ctx.setLineDash([]);
// FTOP
const[fx,fy]=pj(0,0,0);
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=20;
ctx.beginPath();ctx.arc(fx,fy,5,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
if(DV.L){ctx.font='bold '+fs(20)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('\u25c7 FTOP',fx,fy-16)}
// Reference orbits
for(const[rr,col] of [[0.72,'rgba(230,200,120,.55)'],[1.0,'rgba(150,195,255,.65)'],[1.52,'rgba(255,140,100,.55)']]){
ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=1.6;
for(let i=0;i<=48;i++){const a=i*Math.PI*2/48;const[px,py]=pj(rr*Math.cos(a),rr*Math.sin(a),0);
if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}ctx.stroke()}
// Transparent 3D FoL sphere grid (shared helper with independent rotation)
if(DV.X){const to3d=(x,y,z)=>pj(x,y,z*tsc);drawFoL3DSpheres(ctx,to3d,{alpha:0.22,use3D:true,zScale:tsc})}
// Solar system on the z=0 plane.
c._solarHits=drawSolarSystem2D(ctx,(x,y)=>pj(x,y,0),{scale:sc*0.6});
// FoL lattice spheres (same geometry as draw3D) \u2014 green-tinted since this tab is parallel-focused
{const r1=Rs*0.58;const dTx=TG[0],dTy=TG[1],dTr=Math.sqrt(dTx*dTx+dTy*dTy)||1;const ux=dTx/dTr,uy=dTy/dTr;
const flS=[];
flS.push({cx3:0,cy3:0,r:r1,col:'rgba(140,230,170,.55)'});
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;flS.push({cx3:r1*Math.cos(a),cy3:r1*Math.sin(a),r:r1,col:'rgba(120,210,160,.42)'})}
for(let i=0;i<12;i++){const a=i*Math.PI/6;flS.push({cx3:Rs*Math.cos(a),cy3:Rs*Math.sin(a),r:Rs,col:'rgba(230,200,130,.35)'})}
for(let i=0;i<12;i++){const a=(i*30+15)*Math.PI/180;flS.push({cx3:Rs*1.5*Math.cos(a),cy3:Rs*1.5*Math.sin(a),r:Rs,col:'rgba(120,180,200,.25)'})}
const nExt=Math.max(1,Math.floor(dTr/(Rs*0.8)));
for(let L=1;L<=nExt;L++){const dd=L*Rs*0.9;if(dd>dTr+Rs)break;
const cax=ux*dd,cay=uy*dd;const aL=Math.max(0.15,0.40-L*0.04);
for(let i=0;i<6;i++){const a=i*Math.PI/3;
const ox=cax+r1*Math.cos(a),oy=cay+r1*Math.sin(a);
flS.push({cx3:ox,cy3:oy,r:r1,col:'rgba(180,220,150,'+aL+')'})}}
for(const sp of flS){const sr=Math.min(sp.r*0.25,0.12);
for(let ring=0;ring<3;ring++){
ctx.beginPath();ctx.strokeStyle=sp.col;ctx.lineWidth=1.2;
for(let i=0;i<=24;i++){const a=i*Math.PI*2/24;let px,py,pz;
if(ring===0){px=sp.cx3+sr*Math.cos(a);py=sp.cy3+sr*Math.sin(a);pz=0}
else if(ring===1){px=sp.cx3+sr*Math.cos(a);py=sp.cy3;pz=sr*Math.sin(a)}
else{px=sp.cx3;py=sp.cy3+sr*Math.cos(a);pz=sr*Math.sin(a)}
const p=pj(px,py,pz);if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}
ctx.stroke()}
ctx.beginPath();ctx.strokeStyle=sp.col;ctx.lineWidth=1.4;
for(let i=0;i<=36;i++){const a=i*Math.PI*2/36;
const p=pj(sp.cx3+sp.r*Math.cos(a),sp.cy3+sp.r*Math.sin(a),0);
if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}
ctx.stroke()}}
// Tangent spheres at PARALLEL gates (green-tinted)
for(let k=0;k<PAR.gt.length;k++){
const gx=PAR.gt[k][0],gy=PAR.gt[k][1],gz=(k+1)*DTG*tsc,sr=0.15;
const isFib=SAL_FIB.has(k+1);
for(let ring=0;ring<3;ring++){
ctx.beginPath();ctx.strokeStyle='rgba(140,240,180,'+(0.55+k*0.03)+')';ctx.lineWidth=isFib?2.2:1.6;
for(let i=0;i<=32;i++){const a=i*Math.PI*2/32;let px,py,pz;
if(ring===0){px=gx+sr*Math.cos(a);py=gy+sr*Math.sin(a);pz=gz}
else if(ring===1){px=gx+sr*Math.cos(a);py=gy;pz=gz+sr*Math.sin(a)}
else{px=gx;py=gy+sr*Math.cos(a);pz=gz+sr*Math.sin(a)}
const[sx,sy]=pj(px,py,pz);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}
const[sx,sy]=pj(gx,gy,gz);
if(isFib){ctx.save();ctx.shadowColor='#00ff90';ctx.shadowBlur=14;
ctx.beginPath();ctx.arc(sx,sy,18,0,Math.PI*2);ctx.strokeStyle='rgba(80,255,170,.6)';ctx.lineWidth=2.5;ctx.stroke();ctx.restore()}
ctx.beginPath();ctx.arc(sx,sy,14,0,Math.PI*2);ctx.fillStyle='#00dd88';ctx.fill();
ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
ctx.fillStyle='#041810';ctx.font='bold '+fs(22)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(k+1,sx,sy);
ctx.textBaseline='alphabetic';
// Gate \u0394 labels next to the gate marker
const d=PAR.diff[k];
ctx.font=fs(20)+' monospace';ctx.fillStyle='rgba(150,255,180,.85)';ctx.textAlign='left';
ctx.fillText('\u0394pos '+d.d_pos.toFixed(4)+' AU',sx+14,sy-4);
ctx.fillStyle='rgba(170,220,255,.75)';
ctx.fillText('\u0394v '+d.d_vel.toFixed(4)+' AU/yr',sx+14,sy+16);
ctx.fillStyle='rgba(255,220,140,.75)';
ctx.fillText('\u0394hdg '+d.d_hdg.toFixed(2)+'\u00b0'+(isFib?'  F'+(k+1):''),sx+14,sy+34)}
// Tensor mapping lines scope-gate \u2192 parallel-gate (bright green)
ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(80,255,170,.55)';ctx.lineWidth=1.4;
for(let km=0;km<Math.min(GXY.length,PAR.gt.length);km++){
const[sx1,sy1]=pj(GXY[km][0],GXY[km][1],0);const gz2=(km+1)*DTG*tsc;
const[tx1,ty1]=pj(PAR.gt[km][0],PAR.gt[km][1],gz2);
ctx.beginPath();ctx.moveTo(sx1,sy1);ctx.lineTo(tx1,ty1);ctx.stroke()}
ctx.setLineDash([]);
// PRIMARY trajectory \u2014 dimmed reference in 3D
if(DV.T&&NT){ctx.beginPath();ctx.strokeStyle='rgba(255,215,0,.28)';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
for(let i=0;i<NT.length;i++){const tz=i/(NT.length-1)*TF*tsc;const[sx,sy]=pj(NT[i][0],NT[i][1],tz);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([])}
// PARALLEL trajectory \u2014 primary of this tab, speed-colored + glow
if(DV.T){const allSpd=GI.map(g=>g.spd);const mn=Math.min(...allSpd),mx=Math.max(...allSpd),rng=mx-mn+.001;
const col=(s)=>{const t2=(s-mn)/rng;return'rgb('+Math.round(40+t2*80)+','+Math.round(255-t2*40)+','+Math.round(160-t2*60)+')'};
const pnt=PAR.nt;
// Glow
ctx.beginPath();ctx.strokeStyle='rgba(0,255,140,.18)';ctx.lineWidth=9;ctx.lineCap='round';
for(let i=0;i<pnt.length;i++){const tz=i/(pnt.length-1)*TF*tsc;const[sx,sy]=pj(pnt[i][0],pnt[i][1],tz);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}ctx.stroke();
// Speed-colored segments
const nPerSeg=Math.floor(pnt.length/13);
for(let seg=0;seg<13;seg++){const cIdx=Math.min(seg,11);const co=seg<12?col(GI[cIdx].spd):'#44ffaa';
ctx.beginPath();ctx.strokeStyle=co;ctx.lineWidth=3.5;ctx.shadowColor=co;ctx.shadowBlur=8;
const i0=seg*nPerSeg,i1=Math.min((seg+1)*nPerSeg+1,pnt.length);
for(let i=i0;i<i1;i++){const tz=i/(pnt.length-1)*TF*tsc;const[sx,sy]=pj(pnt[i][0],pnt[i][1],tz);if(i===i0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}ctx.shadowBlur=0}
// Primary barrel (dim reference)
{const[bsx,bsy]=pj(B[0],B[1],0);ctx.fillStyle='rgba(255,215,0,.35)';ctx.fillRect(bsx-5,bsy-5,10,10);
ctx.strokeStyle='rgba(255,215,0,.5)';ctx.strokeRect(bsx-5,bsy-5,10,10);
if(DV.L){ctx.font='bold '+fs(22)+' sans-serif';ctx.fillStyle='rgba(255,215,0,.6)';ctx.textAlign='center';ctx.fillText('PRIMARY (ref)',bsx,bsy-14)}}
// PARALLEL barrel \u2014 bright
{const[pbx,pby]=pj(PAR.bp[0],PAR.bp[1],0);ctx.save();ctx.shadowColor='#00ff90';ctx.shadowBlur=18;
ctx.fillStyle='#00dd88';ctx.fillRect(pbx-8,pby-8,16,16);ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(pbx-8,pby-8,16,16);
if(DV.L){ctx.font='bold '+fs(28)+' sans-serif';ctx.fillStyle='#00ff90';ctx.textAlign='center';ctx.fillText('PARALLEL BARREL (+'+PAR.off+'\u00b0)',pbx,pby+28);
ctx.font=fs(22)+' monospace';ctx.fillText('('+PAR.bp[0].toFixed(3)+', '+PAR.bp[1].toFixed(3)+')',pbx,pby+48)}}
// Animated parallel comet
if(animT>0){const f=animT/TF;const idx=f*(PAR.nt.length-1);
const i0=Math.floor(idx),i1=Math.min(i0+1,PAR.nt.length-1);const tt=idx-i0;
const pxP=PAR.nt[i0][0]+(PAR.nt[i1][0]-PAR.nt[i0][0])*tt;
const pyP=PAR.nt[i0][1]+(PAR.nt[i1][1]-PAR.nt[i0][1])*tt;
const az=animT*tsc;const[asx,asy]=pj(pxP,pyP,az);
ctx.save();ctx.shadowColor='#aaffdd';ctx.shadowBlur=22;
ctx.beginPath();ctx.arc(asx,asy,7,0,Math.PI*2);ctx.fillStyle='#ccffee';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(asx,asy,7,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
ctx.font='bold '+fs(24)+' monospace';ctx.fillStyle='#aaffdd';ctx.textAlign='left';
ctx.fillText('T+'+animT.toFixed(3),asx+16,asy)}
// Target at arrival + on ground (supports TGT_Z off-plane)
const tgzP=(TGT_Z||0)*tsc;
const[tsx,tsy]=pj(TG[0],TG[1],TF*tsc+tgzP);
const[tg0x,tg0y]=pj(TG[0],TG[1],0);
ctx.beginPath();ctx.arc(tg0x,tg0y,5,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.3)';ctx.fill();
ctx.beginPath();ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(68,255,136,.25)';ctx.lineWidth=1;
ctx.moveTo(tg0x,tg0y);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([]);
// Vertical z-drop line if off-plane
if(Math.abs(TGT_Z||0)>1e-6){const[tTx,tTy]=pj(TG[0],TG[1],TF*tsc);
ctx.beginPath();ctx.strokeStyle='rgba(136,255,204,.55)';ctx.lineWidth=1.5;ctx.setLineDash([2,3]);
ctx.moveTo(tTx,tTy);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([])}
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=15;
ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.fillStyle='#44ff88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.stroke();
ctx.font='bold '+fs(28)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';ctx.fillText('TARGET',tsx,tsy-26);
ctx.font=fs(22)+' monospace';ctx.fillStyle='#44ff88';
ctx.fillText('('+TG[0].toFixed(2)+', '+TG[1].toFixed(2)+(Math.abs(TGT_Z||0)>1e-6?', '+(TGT_Z||0).toFixed(2):'')+') AU',tsx,tsy+24);
// Title
ctx.font='bold '+fs(30)+' "Times New Roman",serif';ctx.fillStyle='#88ffbb';ctx.textAlign='center';
ctx.fillText('3-D PARALLEL Scope  \u00b7  +'+PAR.off+'\u00b0 offset  \u00b7  \u0394miss '+PAR.miss.toFixed(6)+' AU',w/2,34);
ctx.font=fs(18)+' sans-serif';ctx.fillStyle='#889';
ctx.fillText('Left-drag = rotate  \u00b7  Scroll = zoom  \u00b7  R = reset  \u00b7  Primary shown dim as reference',w/2,h-16);
// Hud
const hud=document.getElementById('p3dHud');
if(hud){const gnow=getAnimGate();const gk=Math.min(gnow-1,11);const dd=PAR.diff[gk];
let hh='<div style="font:bold 11px sans-serif;color:#88ffbb;margin-bottom:6px;border-bottom:1px solid #0a5;padding-bottom:4px">\u25c7 PARALLEL 3D STATUS</div>';
hh+='<div style="display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font-size:9px">';
hh+='<span style="color:#889">Barrel</span><span style="color:#00dd88">('+PAR.bp[0].toFixed(3)+', '+PAR.bp[1].toFixed(3)+')</span>';
hh+='<span style="color:#889">Offset</span><span style="color:#00dd88">+'+PAR.off+'\u00b0</span>';
hh+='<span style="color:#889">Target</span><span style="color:#44ff88">('+TG[0].toFixed(3)+', '+TG[1].toFixed(3)+')</span>';
hh+='<span style="color:#889">Flight</span><span style="color:#ffd700">'+TF+' yr</span>';
hh+='<span style="color:#889">|v\u2080|</span><span style="color:#ffd700">'+Math.sqrt(PAR.v0[0]**2+PAR.v0[1]**2).toFixed(3)+' AU/yr</span>';
hh+='<span style="color:#889">Rotation rx</span><span style="color:#889">'+(p3rx*180/Math.PI).toFixed(1)+'\u00b0</span>';
hh+='<span style="color:#889">Rotation ry</span><span style="color:#889">'+(p3ry*180/Math.PI).toFixed(1)+'\u00b0</span>';
hh+='<span style="color:#889">Zoom</span><span style="color:#889">'+p3z.toFixed(2)+'x</span>';
hh+='<span style="color:#889">\u0394miss</span><span style="color:#00dd88">'+PAR.miss.toExponential(3)+' AU</span>';
hh+='<span style="color:#889">\u0394pos @ gate '+gnow+'</span><span style="color:#00dd88">'+dd.d_pos.toFixed(5)+' AU</span>';
hh+='<span style="color:#889">\u0394vel @ gate '+gnow+'</span><span style="color:#00dd88">'+dd.d_vel.toFixed(5)+' AU/yr</span>';
hh+='<span style="color:#889">\u0394hdg @ gate '+gnow+'</span><span style="color:#00dd88">'+dd.d_hdg.toFixed(2)+'\u00b0</span>';
hh+='</div>';
// Projectile / energy
if(PAR.proj){const pr=PAR.proj;
hh+='<div style="margin-top:6px;border-top:1px solid #0a5;padding-top:4px;font-size:9px">';
hh+='<div style="color:#ffd700;margin-bottom:3px">PROJECTILE / ENERGY</div>';
hh+='<span style="color:#889">\u2300</span> '+pr.diameter_m+' m | <span style="color:#889">m</span> '+pr.mass_kg.toExponential(2)+' kg<br>';
hh+='<span style="color:#889">KE</span> '+pr.ke_j.toExponential(2)+' J ('+pr.tnt_kt.toFixed(2)+' kt)<br>';
hh+='<span style="color:#889">P<sub>avg</sub></span> '+pr.power_avg_W.toExponential(2)+' W<br>';
hh+='<span style="color:#889">\u0394v total</span> '+pr.dv_total_kms.toFixed(2)+' km/s <span style="color:#889">fuel</span> '+(pr.fuel_frac*100).toFixed(2)+'%';
hh+='</div>'}
hud.innerHTML=hh}
// Bottom coord
const coor=document.getElementById('p3dCoord');
if(coor){const gnow=getAnimGate();const dd=PAR.diff[Math.min(gnow-1,11)];
coor.textContent='PARALLEL 3D \u00b7 gate '+gnow+'/12 \u00b7 \u0394pos '+dd.d_pos.toFixed(4)+' AU \u00b7 \u0394v '+dd.d_vel.toFixed(4)+' AU/yr \u00b7 \u0394miss '+PAR.miss.toFixed(6)+' AU'}
// Hover hit data for tab 12 (gates + parallel trajectory samples)
{const gates=[];for(let k=0;k<PAR.gt.length;k++){
const gz=(k+1)*DTG*tsc;const[sx,sy]=pj(PAR.gt[k][0],PAR.gt[k][1],gz);
gates.push({sx:sx,sy:sy,k:k})}
const step=Math.max(1,Math.floor(PAR.nt.length/200));
const traj=[];for(let i=0;i<PAR.nt.length;i+=step){const tz=i/(PAR.nt.length-1)*TF*tsc;
const[sx,sy]=pj(PAR.nt[i][0],PAR.nt[i][1],tz);
traj.push({sx:sx,sy:sy,wx:PAR.nt[i][0],wy:PAR.nt[i][1],ti:i/(PAR.nt.length-1)*TF})}
c._p3dGates=gates;c._p3dTraj=traj}}

function drawImpact(){
drawScat(document.getElementById('bC'),BI,'#ff4444','Baseline \u2014 '+R.bh+'% hit',false);
drawScat(document.getElementById('cC'),CI,'#44ffaa','Corrected \u2014 '+R.ch+'% hit',true)}
function drawScat(c,pts,col,title,isCorrected){
const ctx=c.getContext('2d');const w=c.width=c.parentElement.clientWidth/2-6;const h=c.height=c.parentElement.clientHeight-8;
// Dark gradient background
const bg=ctx.createRadialGradient(w/2,h/2,0,w/2,h/2,w/2);
bg.addColorStop(0,'#0e0e1e');bg.addColorStop(1,'#060612');
ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
// Grid lines
ctx.strokeStyle='rgba(80,70,55,.12)';ctx.lineWidth=0.5;
for(let g=0;g<w;g+=40){ctx.beginPath();ctx.moveTo(g,0);ctx.lineTo(g,h);ctx.stroke()}
for(let g=0;g<h;g+=40){ctx.beginPath();ctx.moveTo(0,g);ctx.lineTo(w,g);ctx.stroke()}
let xn=1e9,xx=-1e9,yn=1e9,yx=-1e9;
for(const[x,y]of pts){if(x<xn)xn=x;if(x>xx)xx=x;if(y<yn)yn=y;if(y>yx)yx=y}
const dx=(xx-xn)*.1+.005,dy=(yx-yn)*.1+.005;xn-=dx;xx+=dx;yn-=dy;yx+=dy;
const sx=w/(xx-xn),sy=h/(yx-yn),sc2=Math.min(sx,sy);
const cx2=w/2,cy2=h/2,mx=(xn+xx)/2,my=(yn+yx)/2;
const ts2=(x,y)=>[cx2+(x-mx)*sc2,cy2-(y-my)*sc2];
const[tgx,tgy]=ts2(TG[0],TG[1]);
// Target zone rings — multiple concentric
for(let rr=1;rr<=3;rr++){circ(ctx,tgx,tgy,HR*sc2*rr,'rgba(68,255,136,'+(0.4/rr)+')',1.5,rr>1?[4,4]:[])}
// Count hits inside HR
let hits=0;for(const[x,y]of pts){const d=Math.sqrt((x-TG[0])**2+(y-TG[1])**2);if(d<HR)hits++}
// Scatter dots — MUCH higher contrast
ctx.save();
for(const[x,y]of pts){const[sx2,sy2]=ts2(x,y);const d=Math.sqrt((x-TG[0])**2+(y-TG[1])**2);
const inZone=d<HR;
ctx.beginPath();ctx.arc(sx2,sy2,inZone?4:3,0,Math.PI*2);
ctx.fillStyle=inZone?'#ffffff':col;ctx.globalAlpha=inZone?1:0.8;ctx.fill();
if(inZone){ctx.shadowColor='#fff';ctx.shadowBlur=8;ctx.beginPath();ctx.arc(sx2,sy2,3,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0}}
ctx.restore();
// Target center dot — bright glow
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=20;
ctx.beginPath();ctx.arc(tgx,tgy,7,0,Math.PI*2);ctx.fillStyle='#44ff88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tgx,tgy,7,0,Math.PI*2);ctx.stroke();
// Crosshair
ctx.strokeStyle='rgba(68,255,136,.3)';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(tgx-30,tgy);ctx.lineTo(tgx+30,tgy);ctx.stroke();
ctx.beginPath();ctx.moveTo(tgx,tgy-30);ctx.lineTo(tgx,tgy+30);ctx.stroke();
// Title
ctx.font='bold '+fs(24)+' "Times New Roman",serif';ctx.fillStyle='#e8d4a8';ctx.textAlign='center';ctx.fillText(title,w/2,34);
// Stats box
ctx.fillStyle='rgba(10,10,26,.85)';ctx.fillRect(w/2-200,40,400,100);ctx.strokeStyle='rgba(90,74,50,.4)';ctx.strokeRect(w/2-200,40,400,100);
ctx.font=fs(20)+' monospace';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
ctx.fillText('Hits: '+hits+'/'+pts.length+' ('+((hits/pts.length)*100).toFixed(1)+'%)',w/2,70);
ctx.fillText('Target: ('+TG[0].toFixed(3)+', '+TG[1].toFixed(3)+') AU',w/2,98);
ctx.fillText('HR: '+HR+' AU | Sims: '+pts.length,w/2,126);
// Axis labels
// Methodology explanation
if(!isCorrected){
ctx.fillStyle='rgba(10,10,26,.88)';ctx.fillRect(12,h-140,w-24,128);
ctx.strokeStyle='rgba(90,74,50,.4)';ctx.strokeRect(12,h-140,w-24,128);
ctx.font='bold '+fs(16)+' "Times New Roman",serif';ctx.fillStyle='#ff6644';ctx.textAlign='left';
ctx.fillText('BASELINE \u2014 No Correction Applied',22,h-112);
ctx.font=fs(12)+' monospace';ctx.fillStyle='#c8a96e';
ctx.fillText('Perturbed initial state: s\u2080 + N(0, \u03c3)',28,h-86);
ctx.fillText('Propagated freely for '+TF+' yr via RK4',28,h-66);
ctx.fillText('No mid-course \u0394v corrections applied',28,h-46);
ctx.fillText('Miss driven by initial state uncertainty',28,h-26)}
if(isCorrected){
ctx.fillStyle='rgba(10,10,26,.88)';ctx.fillRect(12,h-260,w-24,248);
ctx.strokeStyle='rgba(90,74,50,.4)';ctx.strokeRect(12,h-260,w-24,248);
ctx.font='bold '+fs(16)+' "Times New Roman",serif';ctx.fillStyle='#44ffaa';ctx.textAlign='left';
ctx.fillText('\u2699 CORRECTION METHOD: 12-Gate Jacobian \u0394v',22,h-232);
ctx.font=fs(14)+' monospace';ctx.fillStyle='#c8a96e';
ctx.fillText('At each gate k (1\u201312):',22,h-206);
ctx.font=fs(12)+' monospace';ctx.fillStyle='#ffd700';
ctx.fillText('1. Measure state deviation: \u03b4 = (s + noise) \u2212 s\u2080[k]',28,h-184);
ctx.fillText('2. Project miss via Jacobian: \u0394miss = J[k] \u00b7 \u03b4',28,h-164);
ctx.fillText('3. Extract velocity block: Jv = J[k][:,2:4]',28,h-144);
ctx.fillText('4. Solve: \u0394v = Jv\u207b\u00b9 \u00b7 (\u2212\u0394miss)',28,h-124);
ctx.fillText('5. Damp: \u0394v \u00d7= '+DAMP+' \u00d7 (12\u2212k)/12  (gate decay)',28,h-104);
ctx.fillText('6. Clamp: |\u0394v| \u2264 '+MAX_DV+' AU/yr (maneuver limit)',28,h-84);
ctx.fillText('7. Apply: v[k] += \u0394v',28,h-64);
ctx.font=fs(12)+' monospace';ctx.fillStyle='#889';
ctx.fillText('Sensor noise: \u03c3_pos=0.01 AU, \u03c3_vel=0.001 AU/yr',28,h-40);
ctx.fillText('Perturbation: \u03c3=[0.012, 0.012, 0.003, 0.003]',28,h-20)}
ctx.font=fs(18)+' sans-serif';ctx.fillStyle='#889';ctx.textAlign='center';
ctx.fillText('X (AU)',w/2,isCorrected?h-268:h-148);
ctx.save();ctx.translate(14,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('Y (AU)',0,0);ctx.restore()}

function proj(x,y,z,cx,cy,sc){
const x1=x*Math.cos(ry)-z*Math.sin(ry),z1=x*Math.sin(ry)+z*Math.cos(ry);
const y1=y*Math.cos(rx)-z1*Math.sin(rx);
return[cx+x1*sc*z3,cy-y1*sc*z3]}
// Rotation-parameterized variant so tabs with their own rotation state
// (e.g. PARALLEL 3D at tab 12) can reuse draw3D geometry verbatim.
function projR(x,y,z,cx,cy,sc,RX,RY,ZM){
const x1=x*Math.cos(RY)-z*Math.sin(RY),z1=x*Math.sin(RY)+z*Math.cos(RY);
const y1=y*Math.cos(RX)-z1*Math.sin(RX);
return[cx+x1*sc*ZM,cy-y1*sc*ZM]}

// ═══════════════════════════════════════════════════════════════════════
// SHARED: transparent 3D Flower-of-Life sphere lattice.
// The 2D stencil is a cross-section of this 3D lattice. Every tab renders
// the same transparent sphere grid (flower-of-life in 3D) so the reader
// sees how the 2D slice expands into a filled 3D volume. Used by tabs
// 3 (3D VIEW), 7 (SALVO), 8 (TRAJECTORY), 9 (SPHERE NAV), 10 (3D SPHERE).
//   to3d(x,y,z) → [screenX, screenY]  (tab projects AU coords to screen)
//   opts.maxR    clip radius in AU (default: reach across to the target)
//   opts.alpha   base alpha (default 0.12)
//   opts.use3D   draw full spherical wireframe (true) or flat equator only (false)
// ═══════════════════════════════════════════════════════════════════════
function drawFoL3DSpheres(ctx,to3d,opts){
const o=opts||{};const r1=Rs*0.58;
const dTx=TG[0],dTy=TG[1],dTr=Math.sqrt(dTx*dTx+dTy*dTy)||1;
const ux=dTx/dTr,uy=dTy/dTr;
const maxR=o.maxR||Math.max(dTr+Rs,Rs*2);
// Contrast-boosted default: the FoL grid is the *mapping effect* — it must
// read clearly against a black background. Prior default was 0.10 (invisible).
const aB=o.alpha==null?0.24:o.alpha;
const use3D=o.use3D!==false;
// opts.zScale = the tab's extra Z-axis multiplier (e.g. tsc = 2/TF in tab 3/12).
// We pre-divide the sphere ring's Z offset by this so after the caller's
// `z*tsc` the net Z scale is unity — i.e. spheres render as round, not stretched.
const zInv=1/(o.zScale||1);
// Build a 3D sphere lattice that fills space like flower-of-life extended.
// Layer 0: core + 6 petals at FTOP (XY plane).
// Layer 1: 12 scope-ring spheres on XY.
// Layer 2+: axial extrusion — copies of layer 0+1 stacked along ±Z in Rs*0.58 steps.
// Extra layers along FTOP→target axis to fill the corridor.
const spheres=[];
// Base (z=0) layer
spheres.push({x:0,y:0,z:0,r:r1,a:aB*2});
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;
spheres.push({x:r1*Math.cos(a),y:r1*Math.sin(a),z:0,r:r1,a:aB*1.5})}
for(let i=0;i<12;i++){const a=i*Math.PI/6;
spheres.push({x:Rs*Math.cos(a),y:Rs*Math.sin(a),z:0,r:Rs*0.5,a:aB})}
// Vertical extrusion (z layers) — flower-of-life in 3D
if(use3D){
for(let zs=-2;zs<=2;zs++){if(zs===0)continue;
const zc=zs*r1*Math.sqrt(3)/2;
const aZ=aB*(1-Math.abs(zs)*0.25);if(aZ<=0.02)continue;
// staggered offset (hex-close-packing feel)
const ofs=(Math.abs(zs)%2)?r1*0.5:0;
spheres.push({x:ofs,y:0,z:zc,r:r1,a:aZ});
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;
spheres.push({x:ofs+r1*Math.cos(a),y:r1*Math.sin(a),z:zc,r:r1,a:aZ*0.75})}}}
// Corridor extension toward target (on XY)
const nExt=Math.max(1,Math.floor(Math.min(maxR,dTr+Rs)/(Rs*0.9)));
for(let L=1;L<=nExt;L++){
const d=L*Rs*0.9;if(d>maxR)break;
const cax=ux*d,cay=uy*d;
const aL=Math.max(0.03,aB-L*0.015);
for(let i=0;i<6;i++){const a=i*Math.PI/3;
spheres.push({x:cax+r1*Math.cos(a),y:cay+r1*Math.sin(a),z:0,r:r1,a:aL})}}
// Render — farthest first for alpha layering
if(use3D){spheres.sort((p,q)=>(q.x*q.x+q.y*q.y+q.z*q.z)-(p.x*p.x+p.y*p.y+p.z*p.z))}
ctx.save();
// Wider allowed alpha range + thicker line so the mapping effect is visible
// without washing out the trajectory. 3-color axes (warm / cool / green) make
// the three orthogonal rings distinguishable.
for(const s of spheres){
const alpha=Math.max(0.08,Math.min(0.60,s.a));
ctx.lineWidth=Math.max(0.9,alpha*2.2);
// Equatorial circle (XY at the sphere's z-layer) — warm amber
ctx.strokeStyle='rgba(230,200,130,'+alpha+')';
ctx.beginPath();
for(let i=0;i<=36;i++){const a=i*Math.PI*2/36;
const p=to3d(s.x+s.r*Math.cos(a),s.y+s.r*Math.sin(a),s.z*zInv);
if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}
ctx.stroke();
if(use3D){
// XZ ring — cool blue (Z pre-divided by zScale so the caller's z*tsc cancels)
ctx.beginPath();ctx.strokeStyle='rgba(140,190,255,'+(alpha*0.85)+')';
for(let i=0;i<=36;i++){const a=i*Math.PI*2/36;
const p=to3d(s.x+s.r*Math.cos(a),s.y,(s.z+s.r*Math.sin(a))*zInv);
if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}
ctx.stroke();
// YZ ring — green
ctx.beginPath();ctx.strokeStyle='rgba(140,230,170,'+(alpha*0.75)+')';
for(let i=0;i<=36;i++){const a=i*Math.PI*2/36;
const p=to3d(s.x,s.y+s.r*Math.cos(a),(s.z+s.r*Math.sin(a))*zInv);
if(i===0)ctx.moveTo(p[0],p[1]);else ctx.lineTo(p[0],p[1])}
ctx.stroke()}}
ctx.restore();
return spheres.length}

function draw3D(){
const c=document.getElementById('v3C');const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
const cx=w/2,cy=h/2,sc=Math.min(w,h)/4.5*(RS_BASE/Math.max(Rs,1e-9));
ctx.fillStyle='#080810';ctx.fillRect(0,0,w,h);
const tsc=2.0/TF;

// Grid lines on floor — brighter
if(DV.Grid){ctx.strokeStyle='rgba(150,135,105,.45)';ctx.lineWidth=0.9;
for(let g=-2;g<=2;g+=0.5){
const[a1,b1]=proj(g,-2,0,cx,cy,sc);const[a2,b2]=proj(g,2,0,cx,cy,sc);ctx.beginPath();ctx.moveTo(a1,b1);ctx.lineTo(a2,b2);ctx.stroke();
const[c1,d1]=proj(-2,g,0,cx,cy,sc);const[c2,d2]=proj(2,g,0,cx,cy,sc);ctx.beginPath();ctx.moveTo(c1,d1);ctx.lineTo(c2,d2);ctx.stroke()}}
// Axis labels
const[axX,axY]=proj(2.2,0,0,cx,cy,sc);ctx.font='bold '+fs(22)+' monospace';ctx.fillStyle='rgba(200,170,110,.5)';ctx.textAlign='center';ctx.fillText('X (AU)',axX,axY);
const[ayX,ayY]=proj(0,2.2,0,cx,cy,sc);ctx.fillText('Y (AU)',ayX,ayY);
const[azX,azY]=proj(0,0,TF*tsc+0.2,cx,cy,sc);ctx.fillText('T (yr)',azX,azY);
// Time axis line
ctx.beginPath();ctx.strokeStyle='rgba(200,170,110,.3)';ctx.lineWidth=1;ctx.setLineDash([4,6]);
const[z0x,z0y]=proj(0,0,0,cx,cy,sc);const[z1x,z1y]=proj(0,0,TF*tsc,cx,cy,sc);
ctx.moveTo(z0x,z0y);ctx.lineTo(z1x,z1y);ctx.stroke();ctx.setLineDash([]);

// FTOP marker at origin
const[fx,fy]=proj(0,0,0,cx,cy,sc);
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=20;
ctx.beginPath();ctx.arc(fx,fy,5,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
if(DV.L){ctx.font='bold '+fs(20)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('\u25c7 FTOP',fx,fy-16)}

// Reference orbit rings on XY plane (z=0) — brighter so they read against black
for(const[rr,col] of [[0.72,'rgba(230,200,120,.55)'],[1.0,'rgba(150,195,255,.65)'],[1.52,'rgba(255,140,100,.55)']]) {
ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=1.6;
for(let i=0;i<=48;i++){const a=i*Math.PI*2/48;const[px,py]=proj(rr*Math.cos(a),rr*Math.sin(a),0,cx,cy,sc);
if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}ctx.stroke()}

// Transparent 3D Flower-of-Life sphere grid (volumetric background).
// Fills the space around the corridor with a transparent sphere lattice,
// letting the viewer see the 3D expansion of the 2D cross-section template.
if(DV.X){const to3d=(x,y,z)=>proj(x,y,z*tsc,cx,cy,sc);
drawFoL3DSpheres(ctx,to3d,{alpha:0.22,use3D:true,zScale:tsc})}

// Solar system (Sun + 20 planets + moons) on the z=0 plane.
c._solarHits3D=drawSolarSystem2D(ctx,(x,y)=>proj(x,y,0,cx,cy,sc),{scale:sc*0.8,labels:true});

// ═══ 3D FLOWER-OF-LIFE SPHERES — extend outward from FTOP toward TARGET ═══
{const r1=Rs*0.58;
// Direction from FTOP to target
const dTx=TG[0],dTy=TG[1],dTr=Math.sqrt(dTx*dTx+dTy*dTy)||1;
const ux=dTx/dTr,uy=dTy/dTr; // unit vector FTOP→target
const perpX=-uy,perpY=ux; // perpendicular
// Build sphere layers extending from FTOP outward toward target
const flSpheres=[];const flXs=[]; // intersection points
// Layer 0: core (original 7 inner) — brighter so the mapping reads on black
flSpheres.push({cx3:0,cy3:0,r:r1,col:'rgba(140,180,230,.55)',lyr:0});
for(let i=0;i<6;i++){const a=(90+i*60)*Math.PI/180;flSpheres.push({cx3:r1*Math.cos(a),cy3:r1*Math.sin(a),r:r1,col:'rgba(130,160,220,.42)',lyr:0})}
// Layer 1: scope ring (original 12) — amber
for(let i=0;i<12;i++){const a=i*Math.PI/6;flSpheres.push({cx3:Rs*Math.cos(a),cy3:Rs*Math.sin(a),r:Rs,col:'rgba(230,200,130,.35)',lyr:1})}
// Layer 2: outer ring (original 12) — cooler
for(let i=0;i<12;i++){const a=(i*30+15)*Math.PI/180;flSpheres.push({cx3:Rs*1.5*Math.cos(a),cy3:Rs*1.5*Math.sin(a),r:Rs,col:'rgba(120,150,200,.25)',lyr:2})}
// Extended layers: tile along FTOP→target axis with overlapping rings
const nExt=Math.max(1,Math.floor(dTr/(Rs*0.8))); // how many extension steps to reach target
for(let layer=1;layer<=nExt;layer++){
const dist=layer*Rs*0.9; // center offset along axis
if(dist>dTr+Rs)break; // don't go beyond target
const cax=ux*dist,cay=uy*dist;
const lAlpha=Math.max(0.15,0.40-layer*0.04);
// 6 ring around this center
for(let i=0;i<6;i++){const a=i*Math.PI/3;
const ox=cax+r1*Math.cos(a),oy=cay+r1*Math.sin(a);
flSpheres.push({cx3:ox,cy3:oy,r:r1,col:'rgba(210,175,110,'+lAlpha+')',lyr:2+layer})}}
// Compute intersections between adjacent spheres (circle-circle on XY plane)
for(let i=0;i<flSpheres.length;i++){for(let j=i+1;j<flSpheres.length;j++){
const s1=flSpheres[i],s2=flSpheres[j];
const dx=s2.cx3-s1.cx3,dy=s2.cy3-s1.cy3,d=Math.sqrt(dx*dx+dy*dy);
if(d<1e-8||d>s1.r+s2.r||d<Math.abs(s1.r-s2.r))continue;
const a2=(s1.r*s1.r-s2.r*s2.r+d*d)/(2*d);const hh=Math.sqrt(Math.max(0,s1.r*s1.r-a2*a2));
const mx=s1.cx3+a2*dx/d,my=s1.cy3+a2*dy/d;
const px1=mx+hh*(-dy/d),py1=my+hh*(dx/d);
const px2=mx-hh*(-dy/d),py2=my-hh*(dx/d);
// Deduplicate
let dup1=false,dup2=false;
for(const ex of flXs){if(Math.abs(ex[0]-px1)+Math.abs(ex[1]-py1)<0.02)dup1=true;if(Math.abs(ex[0]-px2)+Math.abs(ex[1]-py2)<0.02)dup2=true}
if(!dup1)flXs.push([px1,py1]);if(!dup2)flXs.push([px2,py2])}}
// Sort intersections by distance from FTOP for numbering
flXs.sort((a,b)=>(a[0]*a[0]+a[1]*a[1])-(b[0]*b[0]+b[1]*b[1]));
// Render spheres as wireframe — thicker lines so rings read against black
for(const sp of flSpheres){const sr=Math.min(sp.r*0.25,0.12);
for(let ring=0;ring<3;ring++){
ctx.beginPath();ctx.strokeStyle=sp.col;ctx.lineWidth=1.2;
for(let i=0;i<=24;i++){const a=i*Math.PI*2/24;let px,py,pz;
if(ring===0){px=sp.cx3+sr*Math.cos(a);py=sp.cy3+sr*Math.sin(a);pz=0}
else if(ring===1){px=sp.cx3+sr*Math.cos(a);py=sp.cy3;pz=sr*Math.sin(a)}
else{px=sp.cx3;py=sp.cy3+sr*Math.cos(a);pz=sr*Math.sin(a)}
const[sx2,sy2]=proj(px,py,pz,cx,cy,sc);if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}
ctx.stroke()}
// Equatorial ground circle
ctx.beginPath();ctx.strokeStyle=sp.col;ctx.lineWidth=1.4;
for(let i=0;i<=36;i++){const a=i*Math.PI*2/36;
const[sx2,sy2]=proj(sp.cx3+sp.r*Math.cos(a),sp.cy3+sp.r*Math.sin(a),0,cx,cy,sc);
if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke()}
// Render intersection points using the SHARED XS + FIB_SET defined at the
// top of the script (same as Tab 1 SCOPE uses) \u2014 so every Fibonacci label in
// 3D matches the 2D tab exactly. Both tabs index the same sorted vesica-piscis
// loci and highlight indices 0,1,2,3,5,8,13,21,34,55,89.
const fibPts3=[];
for(let xi=0;xi<XS.length;xi++){
const[ipx,ipy]=XS[xi];const[isx,isy]=proj(ipx,ipy,0,cx,cy,sc);
const isFib=FIB_SET.has(xi);
const xr=Math.sqrt(ipx*ipx+ipy*ipy);const xdeg=((Math.atan2(ipy,ipx)*180/Math.PI)%360+360)%360;
if(isFib){ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=10;
ctx.beginPath();ctx.arc(isx,isy,6,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
ctx.font='bold '+fs(24)+' monospace';ctx.fillStyle='#ffdd44';ctx.textAlign='center';ctx.fillText('F'+xi,isx,isy-14);
ctx.font=fs(16)+' monospace';ctx.fillStyle='rgba(255,215,0,.55)';
ctx.fillText(Math.round(xdeg)+'\u00b0 R:'+xr.toFixed(2),isx,isy+16);
fibPts3.push([isx,isy,xi])}
else{ctx.beginPath();ctx.arc(isx,isy,2.5,0,Math.PI*2);ctx.fillStyle='rgba(220,190,140,.65)';ctx.fill();
ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(220,190,140,.55)';ctx.textAlign='center';ctx.fillText(xi,isx,isy-8)}}
// Fibonacci spiral \u2014 quadratic curve through the Fib points, rendered in 3D
// (projected z=0 so it's on the scope plane, matching Tab 1 SCOPE's spiral).
if(fibPts3.length>1){ctx.beginPath();ctx.strokeStyle='rgba(255,170,60,.55)';ctx.lineWidth=2.5;ctx.setLineDash([6,5]);
ctx.moveTo(fibPts3[0][0],fibPts3[0][1]);
for(let i=1;i<fibPts3.length;i++){
const[x0,y0]=fibPts3[i-1],[x1,y1]=fibPts3[i];
const mxq=(x0+x1)/2+0.15*(y1-y0),myq=(y0+y1)/2-0.15*(x1-x0);
ctx.quadraticCurveTo(mxq,myq,x1,y1)}
ctx.stroke();ctx.setLineDash([])}
// Save counts before block scope closes
c._flSphN=flSpheres.length;c._flXsN=XS.length}

// Wireframe tangent spheres at gates — brighter, with Fibonacci numbering
for(let k=0;k<GT.length;k++){
const gx=GT[k][0],gy=GT[k][1],gz=(k+1)*DTG*tsc,sr=0.15;
const isFib=SAL_FIB.has(k+1);
for(let ring=0;ring<3;ring++){
ctx.beginPath();ctx.strokeStyle='rgba(240,210,150,'+(0.55+k*0.03)+')';ctx.lineWidth=isFib?2.2:1.6;
for(let i=0;i<=32;i++){const a=i*Math.PI*2/32;let px,py,pz;
if(ring===0){px=gx+sr*Math.cos(a);py=gy+sr*Math.sin(a);pz=gz}
else if(ring===1){px=gx+sr*Math.cos(a);py=gy;pz=gz+sr*Math.sin(a)}
else{px=gx;py=gy+sr*Math.cos(a);pz=gz+sr*Math.sin(a)}
const[sx,sy]=proj(px,py,pz,cx,cy,sc);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}
const[sx,sy]=proj(gx,gy,gz,cx,cy,sc);
// Fibonacci gate: larger with glow ring
if(isFib){ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=14;
ctx.beginPath();ctx.arc(sx,sy,18,0,Math.PI*2);ctx.strokeStyle='rgba(255,170,60,.6)';ctx.lineWidth=2.5;ctx.stroke();ctx.restore()}
ctx.beginPath();ctx.arc(sx,sy,14,0,Math.PI*2);ctx.fillStyle='#ffd700';ctx.fill();
ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
ctx.fillStyle='#1a1610';ctx.font='bold '+fs(24)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(k+1,sx,sy);
// Gate label with speed + R + Fibonacci tag
const g=GI[k];ctx.font=fs(24)+' monospace';ctx.fillStyle='rgba(255,215,0,.6)';ctx.textAlign='left';
ctx.fillText(g.spd.toFixed(1)+' AU/yr'+(isFib?' F'+(k+1):''),sx+14,sy-6);ctx.fillStyle='rgba(180,180,180,.5)';
ctx.fillText('R:'+g.rsun.toFixed(2)+' T+'+g.tg.toFixed(2),sx+14,sy+18)}

// Tensor mapping lines — scope gate → trajectory gate (same as Scope tab)
ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(255,215,140,.55)';ctx.lineWidth=1.3;
for(let km=0;km<Math.min(GXY.length,GT.length);km++){
const[sx1,sy1]=proj(GXY[km][0],GXY[km][1],0,cx,cy,sc);const gz2=(km+1)*DTG*tsc;
const[tx1,ty1]=proj(GT[km][0],GT[km][1],gz2,cx,cy,sc);
ctx.beginPath();ctx.moveTo(sx1,sy1);ctx.lineTo(tx1,ty1);ctx.stroke()}
ctx.setLineDash([]);

// Trajectory — speed-colored segments (matching Scope detail)
{const allSpd3=GI.map(g=>g.spd);const mnS3=Math.min(...allSpd3),mxS3=Math.max(...allSpd3),rngS3=mxS3-mnS3+.001;
const spdCol3=(s)=>{const t2=(s-mnS3)/rngS3;return'rgb('+Math.round(80+t2*175)+','+Math.round(255-t2*80)+','+Math.round(200-t2*180)+')'};
const nPerSeg3=Math.floor(NT.length/13);
// Outer glow pass — wider + more visible
ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.35)';ctx.lineWidth=13;ctx.lineCap='round';
for(let i=0;i<NT.length;i++){const tz=i/(NT.length-1)*TF*tsc;const[sx,sy]=proj(NT[i][0],NT[i][1],tz,cx,cy,sc);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}ctx.stroke();ctx.lineCap='butt';
// Speed-colored segments — thicker core + stronger shadow
for(let seg=0;seg<13;seg++){
const colIdx=Math.min(seg,11);const col3s=seg<12?spdCol3(GI[colIdx].spd):'#44ff88';
ctx.beginPath();ctx.strokeStyle=col3s;ctx.lineWidth=5;ctx.shadowColor=col3s;ctx.shadowBlur=10;
const i0=seg*nPerSeg3,i1=Math.min((seg+1)*nPerSeg3+1,NT.length);
for(let i=i0;i<i1;i++){const tz=i/(NT.length-1)*TF*tsc;const[sx,sy]=proj(NT[i][0],NT[i][1],tz,cx,cy,sc);if(i===i0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}ctx.shadowBlur=0}

// Midpoint annotations between gates (curvature + speed change)
for(let k=0;k<GT.length;k++){const g=GI[k];
const prev=k===0?B:GT[k-1];const prevZ=k===0?0:(k)*DTG*tsc;const curZ=(k+1)*DTG*tsc;
const mx2=(prev[0]+GT[k][0])/2,my2=(prev[1]+GT[k][1])/2,mz2=(prevZ+curZ)/2;
const[mpx,mpy]=proj(mx2,my2,mz2,cx,cy,sc);
ctx.font=fs(14)+' sans-serif';ctx.textAlign='center';
ctx.fillStyle='rgba(220,200,140,.55)';ctx.fillText('\u2220'+Math.abs(g.dhdg).toFixed(1)+'\u00b0',mpx,mpy-10);
ctx.fillStyle=g.dspd>=0?'rgba(100,255,150,.5)':'rgba(255,130,100,.5)';
ctx.fillText((g.dspd>=0?'+':'')+g.dspd.toFixed(2)+' AU/yr',mpx,mpy+8)}

// Velocity direction arrows at gates
for(let k=0;k<GT.length;k++){const g=GI[k];const gz=(k+1)*DTG*tsc;
const[gx,gy]=proj(GT[k][0],GT[k][1],gz,cx,cy,sc);
const ang=Math.atan2(-g.vy,g.vx);const len=12;
const ex=gx+len*Math.cos(ang),ey=gy+len*Math.sin(ang);
ctx.beginPath();ctx.moveTo(gx,gy);ctx.lineTo(ex,ey);ctx.strokeStyle='rgba(255,210,60,.6)';ctx.lineWidth=2;ctx.stroke();
const ah=4;ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-ah*Math.cos(ang-.4),ey-ah*Math.sin(ang-.4));
ctx.lineTo(ex-ah*Math.cos(ang+.4),ey-ah*Math.sin(ang+.4));ctx.closePath();ctx.fillStyle='rgba(255,210,60,.6)';ctx.fill()}

// Animated comet in 3D
if(animT>0){const ap=getAnimPos();const az=animT*tsc;
const[asx,asy]=proj(ap[0],ap[1],az,cx,cy,sc);
ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=25;
ctx.beginPath();ctx.arc(asx,asy,7,0,Math.PI*2);ctx.fillStyle='#ffaa00';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(asx,asy,7,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
ctx.font='bold '+fs(28)+' monospace';ctx.fillStyle='#ffaa00';ctx.textAlign='left';
ctx.fillText('T+'+animT.toFixed(3),asx+16,asy);ctx.fillStyle='#fff';ctx.fillText(getAnimSpd().toFixed(2)+' AU/yr',asx+16,asy+28)}

// Barrel
const[bsx,bsy]=proj(B[0],B[1],0,cx,cy,sc);
ctx.fillStyle='#4488ff';ctx.fillRect(bsx-7,bsy-7,14,14);ctx.strokeStyle='#88bbff';ctx.lineWidth=1.5;ctx.strokeRect(bsx-7,bsy-7,14,14);
ctx.font='bold '+fs(32)+' sans-serif';ctx.fillStyle='#66aaff';ctx.textAlign='center';ctx.fillText("BARREL (7 o'clock)",bsx,bsy+30);

// Target (at end time, optionally off-plane via TGT_Z). The z-component is
// rendered in the same projection (Z axis of the 3D scope). A vertical drop
// line connects the target to its ground-plane shadow so off-plane targets
// are legible.
const tgz_vis=(TGT_Z||0)*tsc;  // scope z-units (same mapping as flight time)
const[tsx,tsy]=proj(TG[0],TG[1],TF*tsc+tgz_vis,cx,cy,sc);
const[tg0x,tg0y]=proj(TG[0],TG[1],0,cx,cy,sc);
// Target ground marker + drop line to target-at-end
ctx.beginPath();ctx.arc(tg0x,tg0y,5,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.35)';ctx.fill();
ctx.beginPath();ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(68,255,136,.3)';ctx.lineWidth=1.2;
ctx.moveTo(tg0x,tg0y);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([]);
// If z ≠ 0 draw a vertical z-line from the ground plane to the target height
if(Math.abs(TGT_Z||0)>1e-6){const[tzGx,tzGy]=proj(TG[0],TG[1],TF*tsc,cx,cy,sc);
ctx.beginPath();ctx.strokeStyle='rgba(136,255,204,.55)';ctx.lineWidth=1.5;ctx.setLineDash([2,3]);
ctx.moveTo(tzGx,tzGy);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([])}
// Target at arrival
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=15;
ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.fillStyle='#44ff88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.stroke();
// Crosshair on target
ctx.strokeStyle='rgba(68,255,136,.4)';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(tsx-14,tsy);ctx.lineTo(tsx+14,tsy);ctx.stroke();
ctx.beginPath();ctx.moveTo(tsx,tsy-14);ctx.lineTo(tsx,tsy+14);ctx.stroke();
ctx.font='bold '+fs(32)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';ctx.fillText('TARGET',tsx,tsy-26);
ctx.font=fs(24)+' monospace';ctx.fillStyle='#44ff88';
ctx.fillText('('+TG[0].toFixed(2)+', '+TG[1].toFixed(2)+(Math.abs(TGT_Z||0)>1e-6?', '+(TGT_Z||0).toFixed(2):'')+') AU',tsx,tsy+28);
ctx.fillStyle='#889';ctx.fillText('T+'+TF+' yr'+(Math.abs(TGT_Z||0)>1e-6?'  ·  z='+(TGT_Z||0).toFixed(2)+' AU':''),tsx,tsy+52);

// ═══ TARGET ORBITAL PATH in 3D ═══
if(DV.Tgt&&TGT_ORB){
// History (orange dashed)
if(TGT_ORB.history){ctx.beginPath();ctx.strokeStyle='rgba(255,140,60,.2)';ctx.lineWidth=1;ctx.setLineDash([3,5]);
const hstep=Math.max(1,Math.floor(TGT_ORB.history.length/200));
for(let ih=0;ih<TGT_ORB.history.length;ih+=hstep){const[hx,hy]=proj(TGT_ORB.history[ih][0],TGT_ORB.history[ih][1],0,cx,cy,sc);
if(ih===0)ctx.moveTo(hx,hy);else ctx.lineTo(hx,hy)}ctx.stroke();ctx.setLineDash([])}
// Forward path (green dashed, on ground plane z=0)
if(TGT_ORB.path){ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
const fstep=Math.max(1,Math.floor(TGT_ORB.path.length/200));
for(let ip=0;ip<TGT_ORB.path.length;ip+=fstep){const[fpx,fpy]=proj(TGT_ORB.path[ip][0],TGT_ORB.path[ip][1],0,cx,cy,sc);
if(ip===0)ctx.moveTo(fpx,fpy);else ctx.lineTo(fpx,fpy)}ctx.stroke();ctx.setLineDash([])}
// Target gate-time positions (where target is at each projectile gate time)
if(TGT_ORB.gates){for(let gk=0;gk<TGT_ORB.gates.length;gk++){const gz3t=(gk+1)*DTG*tsc;
const[tgkx,tgky]=proj(TGT_ORB.gates[gk][0],TGT_ORB.gates[gk][1],gz3t,cx,cy,sc);
ctx.beginPath();ctx.arc(tgkx,tgky,3,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.3)';ctx.fill()}}}

// ═══ PARALLEL SCOPE in 3D ═══
if(showParallel&&PAR){
ctx.beginPath();ctx.strokeStyle='rgba(0,220,120,.4)';ctx.lineWidth=1.5;ctx.setLineDash([5,4]);
const pnt3=PAR.nt;for(let i=0;i<pnt3.length;i++){const tz=i/(pnt3.length-1)*TF*tsc;
const[sx2,sy2]=proj(pnt3[i][0],pnt3[i][1],tz,cx,cy,sc);if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}
ctx.stroke();ctx.setLineDash([]);
// Parallel barrel in 3D
const[pb3x,pb3y]=proj(PAR.bp[0],PAR.bp[1],0,cx,cy,sc);
ctx.fillStyle='rgba(0,220,120,.5)';ctx.fillRect(pb3x-4,pb3y-4,8,8);
ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle='#00dd88';ctx.textAlign='center';ctx.fillText('PAR',pb3x,pb3y+18);
// Diff arrows + labeled gate dots in 3D for parallel
for(let k=0;k<PAR.gt.length;k++){
const gz3=(k+1)*DTG*tsc;const[px3,py3]=proj(GT[k][0],GT[k][1],gz3,cx,cy,sc);
const[qx3,qy3]=proj(PAR.gt[k][0],PAR.gt[k][1],gz3,cx,cy,sc);
ctx.beginPath();ctx.moveTo(px3,py3);ctx.lineTo(qx3,qy3);ctx.strokeStyle='rgba(0,255,140,.3)';ctx.lineWidth=1;ctx.stroke();
const pIsFib=SAL_FIB.has(k+1);const pR=pIsFib?4:2.5;
if(pIsFib){ctx.save();ctx.shadowColor='#00ff88';ctx.shadowBlur=6;ctx.beginPath();ctx.arc(qx3,qy3,pR+2,0,Math.PI*2);ctx.strokeStyle='#00dd88';ctx.lineWidth=1.5;ctx.stroke();ctx.restore()}
ctx.beginPath();ctx.arc(qx3,qy3,pR,0,Math.PI*2);ctx.fillStyle='rgba(0,220,120,.6)';ctx.fill();
ctx.font='bold '+fs(20)+' monospace';ctx.fillStyle='#00dd88';ctx.textAlign='center';ctx.fillText((pIsFib?'F':'')+(k+1),qx3,qy3-pR-8)}}

// ═══ MULTI-BARREL SWARM in 3D ═══
if(showSwarm&&SWM){
for(let si=0;si<SWM.length;si++){const sw3=SWM[si];const col3=SWM_COLS[si%SWM_COLS.length];
// Swarm trajectory
ctx.beginPath();ctx.strokeStyle=col3+'66';ctx.lineWidth=1;ctx.setLineDash([3,5]);
const snt3=sw3.nt;for(let i=0;i<snt3.length;i++){const tz=i/(snt3.length-1)*TF*tsc;
const[sx3,sy3]=proj(snt3[i][0],snt3[i][1],tz,cx,cy,sc);if(i===0)ctx.moveTo(sx3,sy3);else ctx.lineTo(sx3,sy3)}
ctx.stroke();ctx.setLineDash([]);
// Swarm barrel
const[sb3x,sb3y]=proj(sw3.bp[0],sw3.bp[1],0,cx,cy,sc);
ctx.beginPath();ctx.arc(sb3x,sb3y,4,0,Math.PI*2);ctx.fillStyle=col3;ctx.fill();
ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle=col3;ctx.textAlign='center';ctx.fillText('B@'+sw3.ck+'h',sb3x,sb3y+16);
// Convergence line to target arrival point
ctx.beginPath();ctx.setLineDash([2,6]);ctx.strokeStyle=col3+'33';ctx.lineWidth=0.7;
ctx.moveTo(sb3x,sb3y);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([]);
// Swarm gate dots — fully labeled
for(let k3=0;k3<sw3.gt.length;k3++){const gz3=(k3+1)*DTG*tsc;
const[sg3x,sg3y]=proj(sw3.gt[k3][0],sw3.gt[k3][1],gz3,cx,cy,sc);
const sIsFib=SAL_FIB.has(k3+1);const sR=sIsFib?3.5:2;
if(sIsFib){ctx.save();ctx.shadowColor=col3;ctx.shadowBlur=5;ctx.beginPath();ctx.arc(sg3x,sg3y,sR+2,0,Math.PI*2);ctx.strokeStyle=col3;ctx.lineWidth=1;ctx.stroke();ctx.restore()}
ctx.beginPath();ctx.arc(sg3x,sg3y,sR,0,Math.PI*2);ctx.fillStyle=col3+'aa';ctx.fill();
ctx.font='bold '+fs(18)+' monospace';ctx.fillStyle=col3;ctx.textAlign='center';ctx.fillText((sIsFib?'F':'')+(k3+1),sg3x,sg3y-sR-6)}}}

// Title
ctx.font='bold '+fs(34)+' "Times New Roman",serif';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
ctx.fillText('3-D Flower-of-Life Scope \u2014 Tangent-Sphere Corridor',w/2,34);
ctx.font=fs(24)+' sans-serif';ctx.fillStyle='#889';
ctx.fillText('31 sphere lattice \u00b7 Left-drag=rotate \u00b7 Scroll=zoom \u00b7 R=reset \u00b7 P=parallel \u00b7 S=swarm \u00b7 Hover=inspect',w/2,h-16);
ctx.fillText('X,Y=position (AU) \u00b7 Z=time (yr) \u00b7 \u25c7=FTOP \u00b7 F1,F2,F3,F5,F8=Fibonacci gates',w/2,h-42);

// Legend — top-left (matching Scope)
// Legend moved to shared clickable #legend overlay (bottom-left fixed panel).

// Hit/miss stats — bottom-right (matching Scope)
ctx.textAlign='right';ctx.font='bold '+fs(20)+' monospace';
ctx.fillStyle='#44ff88';ctx.fillText('Hit:'+R.ch+'% | Miss:'+R.cmm+' AU',w-14,h-66);
ctx.fillStyle='#ff6644';ctx.fillText('Baseline:'+R.bh+'% | Miss:'+R.bmm+' AU',w-14,h-90);
ctx.fillStyle='#667';ctx.font=fs(16)+' sans-serif';ctx.fillText('Gap: '+Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2).toFixed(4)+' AU | |v\u2080|='+Math.sqrt(V0[0]**2+V0[1]**2).toFixed(3)+' AU/yr',w-14,h-112);

// ═══ 3D HUD stats panel (right side) ═══
const v3hud=document.getElementById('v3Hud');
if(v3hud){
let hh3='<div style="font:bold 11px sans-serif;color:#c8a96e;margin-bottom:6px;border-bottom:1px solid #5a4a32;padding-bottom:4px">\u25c7 3D SCOPE STATUS</div>';
hh3+='<div style="display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font-size:9px">';
hh3+='<span style="color:#889">Barrel</span><span style="color:#4488ff">('+B[0].toFixed(3)+', '+B[1].toFixed(3)+') R='+Math.sqrt(B[0]**2+B[1]**2).toFixed(3)+' AU</span>';
hh3+='<span style="color:#889">Target</span><span style="color:#44ff88">('+TG[0].toFixed(3)+', '+TG[1].toFixed(3)+') R='+Math.sqrt(TG[0]**2+TG[1]**2).toFixed(3)+' AU</span>';
hh3+='<span style="color:#889">B\u2192T gap</span><span style="color:#ffd700">'+Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2).toFixed(3)+' AU</span>';
hh3+='<span style="color:#889">Scope Rs</span><span style="color:#c8a96e">'+Rs.toFixed(3)+' AU (\u2300'+(2*Rs).toFixed(3)+', '+(Rs/RS_BASE).toFixed(2)+'\u00d7 base)</span>';
hh3+='<span style="color:#889">Flight time</span><span style="color:#ffd700">'+TF+' yr (dt<sub>g</sub>='+DTG.toFixed(4)+' yr)</span>';
hh3+='<span style="color:#889">|v\u2080|</span><span style="color:#ffd700">'+Math.sqrt(V0[0]**2+V0[1]**2).toFixed(3)+' AU/yr ('+(Math.sqrt(V0[0]**2+V0[1]**2)*1.496e8/3.156e7).toFixed(2)+' km/s)</span>';
hh3+='<span style="color:#889">Gates</span><span style="color:#ffd700">12 \u00b7 FIB {1,2,3,5,8}</span>';
hh3+='<span style="color:#889">\u03bc = GM\u2609</span><span style="color:#889">'+MU.toFixed(4)+' AU\u00b3/yr\u00b2</span>';
hh3+='<span style="color:#889">RK4 step</span><span style="color:#889">0.005 yr \u00b7 err/step \u2248 3\u00d710\u207b\u00b9\u00b3 AU</span>';
hh3+='<span style="color:#889">Rotation rx/ry</span><span style="color:#889">'+( rx*180/Math.PI).toFixed(1)+'\u00b0 / '+(ry*180/Math.PI).toFixed(1)+'\u00b0</span>';
hh3+='<span style="color:#889">Zoom</span><span style="color:#889">'+z3.toFixed(2)+'x</span>';
hh3+='<span style="color:#889">FoL spheres</span><span style="color:#889">'+(c._flSphN||0)+' (7 inner + 12 scope + 12 outer)</span>';
hh3+='<span style="color:#889">XS points</span><span style="color:#ffd700">'+(c._flXsN||0)+' vesica piscis loci</span>';
hh3+='<span style="color:#889">Fib XS</span><span style="color:#ffdd44">F0,F1,F2,F3,F5,F8,F13,F21,F34,F55,F89</span>';
hh3+='<span style="color:#889">Pairs tested</span><span style="color:#889">C('+(c._flSphN||0)+',2) = '+((c._flSphN||0)*((c._flSphN||1)-1)/2)+'</span>';
hh3+='</div>';
// === CAMPAIGN STATISTICS ===
{const bh=parseFloat(R.bh),ch=parseFloat(R.ch);const impr=(ch-bh).toFixed(2);
const factor=(ch/Math.max(bh,0.01)).toFixed(2);
hh3+='<div style="margin-top:6px;border-top:1px solid #3a3020;padding-top:4px;font-size:9px;color:#c8a96e">CAMPAIGN STATS</div>';
hh3+='<div style="display:grid;grid-template-columns:auto 1fr;gap:1px 8px;font-size:9px">';
hh3+='<span style="color:#889">Sims / campaign</span><span style="color:#ffd700">'+R.n.toLocaleString()+'</span>';
hh3+='<span style="color:#889">Baseline hit</span><span style="color:#ff6644">'+R.bh+'%</span>';
hh3+='<span style="color:#889">Corrected hit</span><span style="color:#44ff88">'+R.ch+'%</span>';
hh3+='<span style="color:#889">Improvement</span><span style="color:#ffd700">+'+impr+'pts \u00b7 '+factor+'\u00d7</span>';
hh3+='<span style="color:#889">Miss (baseline)</span><span style="color:#ff6644">'+R.bmm+' AU</span>';
hh3+='<span style="color:#889">Miss (corrected)</span><span style="color:#44ff88">'+R.cmm+' AU</span>';
hh3+='<span style="color:#889">Max miss (B/C)</span><span style="color:#889">'+R.bmx+' / '+R.cmx+' AU</span>';
hh3+='</div>'}
// === TENSOR [T] summary (det, tr, Frobenius) ===
{let trT=0,frob2=0,maxAbs=0;for(let i=0;i<4;i++){trT+=TM[i][i];for(let j=0;j<4;j++){frob2+=TM[i][j]*TM[i][j];maxAbs=Math.max(maxAbs,Math.abs(TM[i][j]))}}
const frobT=Math.sqrt(frob2);
// 4\u00d74 determinant via cofactor expansion
const det4=((m)=>{const a=m[0],b=m[1],c=m[2],d=m[3];return a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2])-b[2]*(c[1]*d[3]-c[3]*d[1])+b[3]*(c[1]*d[2]-c[2]*d[1]))-a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2])-b[2]*(c[0]*d[3]-c[3]*d[0])+b[3]*(c[0]*d[2]-c[2]*d[0]))+a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1])-b[1]*(c[0]*d[3]-c[3]*d[0])+b[3]*(c[0]*d[1]-c[1]*d[0]))-a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1])-b[1]*(c[0]*d[2]-c[2]*d[0])+b[2]*(c[0]*d[1]-c[1]*d[0]))})(TM);
hh3+='<div style="margin-top:6px;border-top:1px solid #3a3020;padding-top:4px;font-size:9px;color:#c8a96e">RELATIVE TENSOR [T]</div>';
hh3+='<div style="display:grid;grid-template-columns:auto 1fr;gap:1px 8px;font-size:9px">';
hh3+='<span style="color:#889">det(T)</span><span style="color:#ffd700">'+det4.toFixed(4)+'</span>';
hh3+='<span style="color:#889">tr(T)</span><span style="color:#ffd700">'+trT.toFixed(4)+'</span>';
hh3+='<span style="color:#889">\u2016T\u2016<sub>F</sub></span><span style="color:#ffd700">'+frobT.toFixed(4)+'</span>';
hh3+='<span style="color:#889">max|T\u1d62\u2c7c|</span><span style="color:#ffd700">'+maxAbs.toFixed(4)+'</span>';
hh3+='<span style="color:#889">T<sub>xv<sub>y</sub></sub></span><span style="color:#889">'+TM[0][3].toFixed(3)+' (pos\u2194vel coupling)</span>';
hh3+='<span style="color:#889">T<sub>yv<sub>x</sub></sub></span><span style="color:#889">'+TM[1][2].toFixed(3)+'</span>';
hh3+='</div>'}
// === PARALLEL sensitivity block (always visible when PAR exists) ===
if(PAR){const gk=Math.min(getAnimGate()-1,11);const d=PAR.diff[gk];
hh3+='<div style="margin-top:6px;border-top:1px solid #0a5;padding-top:4px;font-size:9px;color:#00dd88">PARALLEL \u0394 (+'+PAR.off+'\u00b0)</div>';
hh3+='<div style="display:grid;grid-template-columns:auto 1fr;gap:1px 8px;font-size:9px">';
hh3+='<span style="color:#889">\u0394miss</span><span style="color:#00dd88">'+PAR.miss.toExponential(2)+' AU</span>';
hh3+='<span style="color:#889">\u0394pos @ gate</span><span style="color:#00dd88">'+d.d_pos.toFixed(5)+' AU</span>';
hh3+='<span style="color:#889">\u0394vel @ gate</span><span style="color:#00dd88">'+d.d_vel.toFixed(5)+' AU/yr</span>';
hh3+='<span style="color:#889">\u0394hdg @ gate</span><span style="color:#00dd88">'+d.d_hdg.toFixed(2)+'\u00b0</span>';
hh3+='</div>'}
hh3+='<div style="margin-top:8px;border-top:1px solid #3a3020;padding-top:6px;font-size:9px">';
hh3+='<div style="color:#c8a96e;margin-bottom:3px">GATE SPEED · J-COND · ΔHDG</div>';
const gspdMx=Math.max(...GI.map(g=>g.spd));
const gjcMx=Math.max(...GI.map(g=>g.jc));
const gdhMx=Math.max(...GI.map(g=>Math.abs(g.dhdg)))||1;
for(const g of GI){const pct=g.spd/gspdMx*100;const isFib=SAL_FIB.has(g.n);
const jpct=g.jc/gjcMx*100;const dhpct=Math.abs(g.dhdg)/gdhMx*100;
hh3+='<div style="display:flex;align-items:center;gap:3px;margin:1px 0">';
hh3+='<span style="color:'+(isFib?'#ffd700':'#667')+';font-size:8px;width:14px;text-align:right">'+(isFib?'F':'')+g.n+'</span>';
hh3+='<div style="flex:1;height:6px;background:#1a1610;border-radius:2px"><div style="width:'+pct+'%;height:100%;background:'+(isFib?'#ffd700':'#44aa88')+';border-radius:2px"></div></div>';
hh3+='<span style="color:#889;font-size:8px;width:28px;text-align:right">'+g.spd.toFixed(1)+'</span>';
hh3+='<div style="flex:0 0 36px;height:6px;background:#1a1610;border-radius:2px"><div style="width:'+jpct+'%;height:100%;background:#88aaff;border-radius:2px"></div></div>';
hh3+='<span style="color:#88aaff;font-size:8px;width:22px;text-align:right">'+g.jc.toFixed(1)+'</span>';
hh3+='<div style="flex:0 0 30px;height:6px;background:#1a1610;border-radius:2px"><div style="width:'+dhpct+'%;height:100%;background:'+(g.dhdg>=0?'#44ff88':'#ff6644')+';border-radius:2px"></div></div>';
hh3+='<span style="color:#889;font-size:8px;width:24px;text-align:right">'+g.dhdg.toFixed(0)+'°</span>';
hh3+='</div>'}
hh3+='<div style="font-size:8px;color:#556;margin-top:2px">green=spd · blue=J-cond · green/red=Δhdg</div>';
hh3+='</div>';
if(animT>0){hh3+='<div style="margin-top:6px;border-top:1px solid #3a3020;padding-top:4px;font-size:9px">';
hh3+='<div style="color:#ffaa00">&#9790; T+'+animT.toFixed(3)+' yr | Gate '+getAnimGate()+'/12</div>';
hh3+='<div style="color:#ffd700">Speed: '+getAnimSpd().toFixed(3)+' AU/yr</div></div>'}
v3hud.innerHTML=hh3}

// Store gate screen positions for 3D hover
const _v3Gates=[];for(let k=0;k<GT.length;k++){
const gz=( k+1)*DTG*tsc;const[sx3,sy3]=proj(GT[k][0],GT[k][1],gz,cx,cy,sc);
_v3Gates.push({sx:sx3,sy:sy3,k:k,g:GI[k]})}
// Store trajectory screen points for hover
const _v3Traj=[];const _v3Step=Math.max(1,Math.floor(NT.length/300));
for(let i=0;i<NT.length;i+=_v3Step){const tz=i/(NT.length-1)*TF*tsc;
const[sx3,sy3]=proj(NT[i][0],NT[i][1],tz,cx,cy,sc);
_v3Traj.push({sx:sx3,sy:sy3,wx:NT[i][0],wy:NT[i][1],t:i/(NT.length-1)*TF})}
const v3c=document.getElementById('v3C');
v3c._gates=_v3Gates;v3c._traj=_v3Traj;

// 3D mouse controls
v3c.onmousedown=e=>{if(e.button===0){drg=true;lmx=e.clientX;lmy=e.clientY}};
v3c.onmouseup=()=>drg=false;v3c.onmouseleave=()=>{drg=false;document.getElementById('v3Tip').style.display='none';document.getElementById('v3Coord').textContent=''};
v3c.onmousemove=e=>{
const rc3=v3c.getBoundingClientRect();const mx3=e.clientX-rc3.left,my3=e.clientY-rc3.top;
if(drg){ry+=(e.clientX-lmx)*0.008;rx+=(e.clientY-lmy)*0.008;lmx=e.clientX;lmy=e.clientY;draw3D();return}
// Hover: check gates first (priority), then trajectory
const tip3=document.getElementById('v3Tip');const coord3=document.getElementById('v3Coord');
const gates3=v3c._gates||[];let found3=null,bestD3=1600;
for(const g of gates3){const dx=mx3-g.sx,dy=my3-g.sy;const d=dx*dx+dy*dy;if(d<bestD3){bestD3=d;found3={type:'gate',g:g.g,k:g.k}}}
if(!found3||bestD3>400){const traj3=v3c._traj||[];for(const p of traj3){const dx=mx3-p.sx,dy=my3-p.sy;const d=dx*dx+dy*dy;if(d<bestD3){bestD3=d;found3={type:'traj',p:p}}}}
if(found3&&bestD3<900){
if(found3.type==='gate'){const g=found3.g;const isFib=SAL_FIB.has(found3.k+1);
const sa=sphereAddr(g.pos[0],g.pos[1]);
const dTgtG=Math.sqrt((g.pos[0]-TG[0])**2+(g.pos[1]-TG[1])**2);
const dBarG=Math.sqrt((g.pos[0]-B[0])**2+(g.pos[1]-B[1])**2);
const pgg=g.vx*g.vx+g.vy*g.vy;const vMag=Math.sqrt(pgg);
tip3.innerHTML='<b style="color:#ffd700">'+(isFib?'\u2605 Fibonacci ':'')+'Gate '+(found3.k+1)+'</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+g.pos[0]+', '+g.pos[1]+') AU</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">Speed:</span> <span style="color:#ffd700">'+g.spd+' AU/yr</span> ('+(vMag*1.496e8/3.156e7).toFixed(2)+' km/s)<br>'+
'<span style="color:#889">v:</span> <span style="color:#ffd700">('+g.vx+', '+g.vy+')</span> AU/yr<br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+g.rsun+' AU</span><br>'+
'<span style="color:#889">Heading:</span> <span style="color:#ffd700">'+g.hdg+'\u00b0</span> \u00b7 <span style="color:#889">\u0394hdg:</span> <span style="color:'+(g.dhdg>=0?'#44ff88':'#ff8866')+'">'+(g.dhdg>=0?'+':'')+g.dhdg+'\u00b0</span><br>'+
'<span style="color:#889">\u0394spd:</span> <span style="color:'+(g.dspd>=0?'#44ff88':'#ff8866')+'">'+(g.dspd>=0?'+':'')+g.dspd+'</span> AU/yr<br>'+
'<span style="color:#889">Curvature:</span> <span style="color:#ffd700">'+g.curv+' /AU</span><br>'+
'<span style="color:#889">Arc len:</span> <span style="color:#ffd700">'+g.arc+' AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+g.tg+' yr</span> (t<sub>go</sub> '+g.tr+')<br>'+
'<span style="color:#889">J-cond:</span> <span style="color:#ffd700">'+g.jc+'</span><br>'+
'<span style="color:#889">gravity:</span> <span style="color:#ffd700">'+g.grav+' AU/yr\u00b2</span><br>'+
'<span style="color:#889">d(Barrel):</span> <span style="color:#ffd700">'+dBarG.toFixed(4)+'</span> \u00b7 <span style="color:#889">d(Target):</span> <span style="color:#44ff88">'+dTgtG.toFixed(4)+'</span>';
coord3.textContent='Gate '+(found3.k+1)+' \u2022 '+sa.addr+' \u2022 '+g.spd+' AU/yr \u2022 T+'+g.tg+'yr'}
else{const p=found3.p;const rr3=Math.sqrt(p.wx*p.wx+p.wy*p.wy);
const dTgt3=Math.sqrt((p.wx-TG[0])**2+(p.wy-TG[1])**2);
const sa=sphereAddr(p.wx,p.wy);
tip3.innerHTML='<b style="color:#44ff88">Trajectory sample</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+p.wx.toFixed(4)+', '+p.wy.toFixed(4)+') AU</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+rr3.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">D(Target):</span> <span style="color:'+(dTgt3<0.5?'#44ff88':'#ffd700')+'">'+dTgt3.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+p.t.toFixed(3)+' yr</span> ('+((p.t/TF)*100).toFixed(1)+'%)';
coord3.textContent=sa.addr+' \u2022 ('+p.wx.toFixed(3)+', '+p.wy.toFixed(3)+') \u2022 T+'+p.t.toFixed(3)+'yr \u2022 D:'+dTgt3.toFixed(3)}
tip3.style.display='block';tip3.style.left=Math.min(mx3+16,rc3.width-320)+'px';tip3.style.top=Math.max(my3-100,10)+'px'}
else{tip3.style.display='none';coord3.textContent=''}};
v3c.onwheel=e=>{z3*=e.deltaY>0?0.9:1.1;z3=Math.max(0.3,Math.min(4,z3));draw3D();e.preventDefault()}}

// PARALLEL 3D (tab 12) mouse controls — rotation + zoom + hover tooltip
{const p3c=document.getElementById('p3dC');const p3tip=document.getElementById('p3dTip');if(p3c){
p3c.onmousedown=e=>{if(e.button===0){p3drg=true;p3lmx=e.clientX;p3lmy=e.clientY}};
p3c.onmouseup=()=>p3drg=false;
p3c.onmouseleave=()=>{p3drg=false;if(p3tip)p3tip.style.display='none'};
p3c.onmousemove=e=>{
if(p3drg){p3ry+=(e.clientX-p3lmx)*0.008;p3rx+=(e.clientY-p3lmy)*0.008;p3lmx=e.clientX;p3lmy=e.clientY;drawParallel3D();return}
if(!p3tip)return;
const rc=p3c.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const gates=p3c._p3dGates||[];let found=null,bestD=1600;
for(const g of gates){const dx=mx-g.sx,dy=my-g.sy;const d=dx*dx+dy*dy;if(d<bestD){bestD=d;found={type:'pgate',k:g.k}}}
if(!found||bestD>400){const trj=p3c._p3dTraj||[];for(const p of trj){const dx=mx-p.sx,dy=my-p.sy;const d=dx*dx+dy*dy;if(d<bestD){bestD=d;found={type:'ptraj',p:p}}}}
if(found&&bestD<900){let html='';
if(found.type==='pgate'){const k=found.k;const d=PAR.diff[k];const isFib=SAL_FIB.has(k+1);
const sa=sphereAddr(PAR.gt[k][0],PAR.gt[k][1]);
html='<b style="color:#00ff90">'+(isFib?'★ Fib ':'')+'Parallel Gate '+(k+1)+'</b><br>'+
'<span style="color:#889">Pos:</span> <span style="color:#44ffaa">('+PAR.gt[k][0].toFixed(4)+', '+PAR.gt[k][1].toFixed(4)+')</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">Δpos:</span> <span style="color:#00dd88">'+d.d_pos.toFixed(5)+' AU</span><br>'+
'<span style="color:#889">Δvel:</span> <span style="color:#00dd88">'+d.d_vel.toFixed(5)+' AU/yr</span><br>'+
'<span style="color:#889">Δhdg:</span> <span style="color:#00dd88">'+d.d_hdg.toFixed(3)+'°</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+((k+1)*DTG).toFixed(4)+' yr</span>'}
else{const p=found.p;const rr=Math.sqrt(p.wx*p.wx+p.wy*p.wy);
const dTgt=Math.sqrt((p.wx-TG[0])**2+(p.wy-TG[1])**2);
const sa=sphereAddr(p.wx,p.wy);
html='<b style="color:#88ffcc">Parallel Trajectory</b><br>'+
'<span style="color:#889">Pos:</span> <span style="color:#44ffaa">('+p.wx.toFixed(4)+', '+p.wy.toFixed(4)+')</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+rr.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">D(Target):</span> <span style="color:'+(dTgt<0.5?'#44ff88':'#ffd700')+'">'+dTgt.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+p.ti.toFixed(3)+' yr</span>'}
p3tip.innerHTML=html;p3tip.style.display='block';
p3tip.style.left=Math.min(mx+16,p3c.clientWidth-330)+'px';p3tip.style.top=Math.max(my-100,10)+'px'}
else{p3tip.style.display='none'}};
p3c.onwheel=e=>{p3z*=e.deltaY>0?0.9:1.1;p3z=Math.max(0.3,Math.min(4,p3z));drawParallel3D();e.preventDefault()}}}

// Scope mouse: tooltip + zoom + AU coordinates + right-click pan + planet-click target
const sC=document.getElementById('sC');const tip=document.getElementById('tip');const coordEl=document.getElementById('coord');
sC.oncontextmenu=e=>e.preventDefault();
// Left-click a planet / moon / sun to set it as the TARGET (triggers /resim).
sC.addEventListener('click',e=>{
 if(e.button!==0)return;
 const rc=sC.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
 const hits=sC._solarHits||[];let pick=null,bestD=9999;
 for(const bh of hits){const dx=mx-bh.x,dy=my-bh.y;const d=dx*dx+dy*dy;if(d<bh.r*bh.r&&d<bestD){bestD=d;pick=bh}}
 if(!pick)return;
 // POST planet_idx / moon_idx to /resim — the backend resolves the actual (x,y).
 const bd={bx:B[0],by:B[1],tx:TG[0],ty:TG[1],hr:HR,sims:R.n,tf:TF,
  swarm_clocks:[1,3,5,9,11],swarm_positions:[null,null,null,null,null],
  planet_idx:pick.planet_idx,moon_idx:pick.moon_idx>=0?pick.moon_idx:null};
 fetch('/resim',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(bd)})
  .then(r=>r.json()).then(d=>{if(d.ok)location.reload()})});
sC.onmousedown=e=>{if(e.button===2){panDrg=true;lmx=e.clientX;lmy=e.clientY;e.preventDefault()}};
sC.onmouseup=e=>{if(e.button===2)panDrg=false};
sC.onmousemove=e=>{
if(panDrg){
// Pan in world coords (Simulation.py: cam -= dx/zoom)
const baseSc=Math.min(sC.width||800,sC.height||600)/5.6*(RS_BASE/Math.max(Rs,1e-9));const sc=baseSc*camZ;
const dx=e.clientX-lmx,dy=e.clientY-lmy;
camX-=dx/sc;camY+=dy/sc;  // += for Y because screen Y is inverted
lmx=e.clientX;lmy=e.clientY;drawScope();return}
const rc=sC.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
// Inverse transform: screen → AU world coords (Simulation.py pattern)
const baseSc=Math.min(sC.width||800,sC.height||600)/5.6*(RS_BASE/Math.max(Rs,1e-9));const sc=baseSc*camZ;
const cxP=sC.width/2,cyP=sC.height/2;
const auX=(mx-cxP)/sc+camX,auY=-(my-cyP)/sc+camY;
const rFtop=Math.sqrt(auX*auX+auY*auY);
const dTgt=Math.sqrt((auX-TG[0])**2+(auY-TG[1])**2);
coordEl.innerHTML='('+auX.toFixed(3)+', '+auY.toFixed(3)+') AU | R<sub>FTOP</sub>: '+rFtop.toFixed(3)+' AU | d<sub>TGT</sub>: '+dTgt.toFixed(3)+' AU | Zoom: '+camZ.toFixed(camZ<1?3:1)+'x';
mouseAU=[auX,auY];
let found=null;let best=999;
for(const p of hPts){const dx=mx-p.sx,dy=my-p.sy,d=dx*dx+dy*dy;if(d<20*20&&d<best){best=d;found=p}}
if(found){tip.style.display='block';tip.style.left=Math.min(e.clientX+12,window.innerWidth-270)+'px';tip.style.top=Math.min(e.clientY-10,window.innerHeight-200)+'px';tip.innerHTML=found.h}else{tip.style.display='none'}};
sC.onmouseleave=()=>{tip.style.display='none';coordEl.innerHTML='';panDrg=false};
// Zoom toward mouse (Simulation.py: preserve world point under cursor)
sC.onwheel=e=>{
const rc=sC.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const baseSc=Math.min(sC.width||800,sC.height||600)/5.6*(RS_BASE/Math.max(Rs,1e-9));
const oldZ=camZ;const oldSc=baseSc*oldZ;
const cxP=sC.width/2,cyP=sC.height/2;
// World point under mouse BEFORE zoom
const wMx=(mx-cxP)/oldSc+camX,wMy=-(my-cyP)/oldSc+camY;
// Apply zoom
camZ*=e.deltaY>0?0.88:1.14;camZ=Math.max(MIN_ZOOM,Math.min(MAX_ZOOM,camZ));
const newSc=baseSc*camZ;
// Adjust camera so world point stays under mouse AFTER zoom
camX=wMx-(mx-cxP)/newSc;camY=wMy+(my-cyP)/newSc;
drawScope();e.preventDefault()};

// === PARALLEL tab pan + zoom + hover tooltip ===
{const pC=document.getElementById('parC');const pTip=document.getElementById('parTip');if(pC){
pC.oncontextmenu=e=>e.preventDefault();
pC.onmousedown=e=>{if(e.button===2){parPanDrg=true;parLmx=e.clientX;parLmy=e.clientY;e.preventDefault()}};
pC.onmouseup=e=>{if(e.button===2)parPanDrg=false};
pC.onmouseleave=()=>{parPanDrg=false;if(pTip)pTip.style.display='none'};
pC.onmousemove=e=>{
if(parPanDrg){const baseSc=Math.min(pC.width||800,pC.height||600)/5.6*(RS_BASE/Math.max(Rs,1e-9));const sc=baseSc*parCamZ;
parCamX-=(e.clientX-parLmx)/sc;parCamY+=(e.clientY-parLmy)/sc;
parLmx=e.clientX;parLmy=e.clientY;drawParallel();return}
// Hover feature lookup
if(!pTip)return;
const rc=pC.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const hits=pC._parHits||[];let found=null,bestD=9999;
for(const h of hits){const dx=mx-h.sx,dy=my-h.sy,d=dx*dx+dy*dy;if(d<h.r*h.r&&d<bestD){bestD=d;found=h}}
if(!found){const trj=pC._parTraj||[];for(const p of trj){const dx=mx-p.sx,dy=my-p.sy,d=dx*dx+dy*dy;if(d<100&&d<bestD){bestD=d;found={type:'ptraj',p:p}}}}
if(found){let html='';
if(found.type==='pgate'){const k=found.k;const d=PAR.diff[k];const isFib=SAL_FIB.has(k+1);
const sa=sphereAddr(PAR.gt[k][0],PAR.gt[k][1]);
html='<b style="color:#00ff90">'+(isFib?'★ Fib ':'')+'Parallel Gate '+(k+1)+'</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#44ffaa">('+PAR.gt[k][0].toFixed(4)+', '+PAR.gt[k][1].toFixed(4)+')</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">Δpos:</span> <span style="color:#00dd88">'+d.d_pos.toFixed(5)+' AU</span><br>'+
'<span style="color:#889">Δvel:</span> <span style="color:#00dd88">'+d.d_vel.toFixed(5)+' AU/yr</span><br>'+
'<span style="color:#889">Δhdg:</span> <span style="color:#00dd88">'+d.d_hdg.toFixed(3)+'°</span><br>'+
'<span style="color:#889">Δx, Δy:</span> <span style="color:#ffd700">('+d.dx+', '+d.dy+')</span><br>'+
'<span style="color:#889">Δvx, Δvy:</span> <span style="color:#ffd700">('+d.dvx+', '+d.dvy+')</span>'}
else if(found.type==='pbar'){const vm=Math.sqrt(PAR.v0[0]*PAR.v0[0]+PAR.v0[1]*PAR.v0[1]);
html='<b style="color:#00ff90">PARALLEL BARREL</b><br>'+
'<span style="color:#889">Offset:</span> <span style="color:#00dd88">+'+PAR.off+'°</span><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+PAR.bp[0].toFixed(4)+', '+PAR.bp[1].toFixed(4)+') AU</span><br>'+
'<span style="color:#889">|v₀|:</span> <span style="color:#ffd700">'+vm.toFixed(4)+' AU/yr</span><br>'+
'<span style="color:#889">Miss:</span> <span style="color:#44ff88">'+PAR.miss.toExponential(3)+' AU</span>';
if(PAR.proj){const pr=PAR.proj;html+='<br><span style="color:#889">⌀:</span> <span style="color:#ffd700">'+pr.diameter_m+' m</span> | <span style="color:#889">m:</span> <span style="color:#ffd700">'+pr.mass_kg.toExponential(2)+' kg</span>';
html+='<br><span style="color:#889">KE:</span> <span style="color:#ffaa44">'+pr.ke_j.toExponential(2)+' J</span> ('+pr.tnt_kt.toFixed(2)+' kt)';
html+='<br><span style="color:#889">P<sub>avg</sub>:</span> <span style="color:#ffaa44">'+pr.power_avg_W.toExponential(2)+' W</span>'}}
else if(found.type==='primbar'){html='<b style="color:#ffd700">PRIMARY BARREL (ref)</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+B[0].toFixed(4)+', '+B[1].toFixed(4)+') AU</span><br>'+
'<span style="color:#889">|v₀|:</span> <span style="color:#ffd700">'+Math.sqrt(V0[0]*V0[0]+V0[1]*V0[1]).toFixed(4)+' AU/yr</span>'}
else if(found.type==='tgt'){html='<b style="color:#44ff88">TARGET</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+TG[0].toFixed(4)+', '+TG[1].toFixed(4)+') AU</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+Math.sqrt(TG[0]*TG[0]+TG[1]*TG[1]).toFixed(4)+' AU</span><br>'+
'<span style="color:#889">Hit radius:</span> <span style="color:#44ff88">'+HR+' AU</span>'}
else if(found.type==='ptraj'){const p=found.p;const rr=Math.sqrt(p.wx*p.wx+p.wy*p.wy);
const dTgt=Math.sqrt((p.wx-TG[0])**2+(p.wy-TG[1])**2);
const sa=sphereAddr(p.wx,p.wy);
html='<b style="color:#88ffcc">Parallel Trajectory</b><br>'+
'<span style="color:#889">Pos:</span> <span style="color:#44ffaa">('+p.wx.toFixed(4)+', '+p.wy.toFixed(4)+') AU</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+rr.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">D(Target):</span> <span style="color:'+(dTgt<0.5?'#44ff88':'#ffd700')+'">'+dTgt.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+p.ti.toFixed(3)+' yr</span>'}
pTip.innerHTML=html;pTip.style.display='block';
pTip.style.left=Math.min(e.clientX+14,window.innerWidth-330)+'px';
pTip.style.top=Math.min(e.clientY-10,window.innerHeight-200)+'px'}
else{pTip.style.display='none'}};
pC.onwheel=e=>{
const rc=pC.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const baseSc=Math.min(pC.width||800,pC.height||600)/5.6*(RS_BASE/Math.max(Rs,1e-9));
const oldZ=parCamZ;const oldSc=baseSc*oldZ;
const cxP=pC.width/2,cyP=pC.height/2;
const wMx=(mx-cxP)/oldSc+parCamX,wMy=-(my-cyP)/oldSc+parCamY;
parCamZ*=e.deltaY>0?0.88:1.14;parCamZ=Math.max(MIN_ZOOM,Math.min(MAX_ZOOM,parCamZ));
const newSc=baseSc*parCamZ;
parCamX=wMx-(mx-cxP)/newSc;parCamY=wMy+(my-cyP)/newSc;
drawParallel();e.preventDefault()}}}

// Stats page
(function(){const sp=document.getElementById('st-pg');let h='';
// ═══ EXTREME TOTAL SYSTEM OVERVIEW ═══
// One dashboard showing everything the simulation knows at a glance.
// Top strip: big-number KPIs. Then dense cards for each subsystem.
const grow=(Rs/(RS_BASE||1.2));
const v0mag=Math.sqrt(V0[0]*V0[0]+V0[1]*V0[1]);
const totalArc=GI.reduce((a,g)=>a+g.arc,0);
const totalTurn=GI.reduce((a,g)=>a+Math.abs(g.dhdg),0);
const avgSpd=GI.reduce((a,g)=>a+g.spd,0)/GI.length;
const totalDvGates=GI.reduce((a,g)=>a+Math.abs(g.dspd),0);
const nBarrels=1+(PAR?1:0)+SWM.length;
// Combined hit probability: baseline per barrel ~99% corrected, use R.ch/100 as p
const pPer=R.ch/100;const pCombined=1-Math.pow(1-pPer,nBarrels);
const sumKE=(ENER&&ENER.primary_proj)?ENER.primary_proj.ke_j*nBarrels:0;
const sumPwr=(ENER&&ENER.primary_proj)?ENER.primary_proj.power_avg_W*nBarrels:0;
// --- KPI STRIP ---
const kpi=(lbl,val,col)=>'<div style="flex:1;min-width:130px;background:linear-gradient(180deg,#1e1812,#12100c);border:1px solid '+(col||'#3a3020')+';border-radius:6px;padding:10px 12px"><div style="font:9px sans-serif;color:#889;letter-spacing:1px">'+lbl+'</div><div style="font:bold 20px monospace;color:'+(col||'#ffd700')+';margin-top:2px">'+val+'</div></div>';
h+='<div style="display:flex;gap:8px;flex-wrap:wrap;width:100%;max-width:none">';
h+=kpi('CORRECTED HIT',R.ch+'%','#44ff88');
h+=kpi('BASELINE HIT',R.bh+'%','#ff6644');
h+=kpi('IMPROVEMENT',(R.ch/Math.max(R.bh,0.01)).toFixed(1)+'×','#ffd700');
h+=kpi('CORRECTED MISS',R.cmm+' AU','#88ffcc');
h+=kpi('BARRELS',nBarrels+'','#cc99ff');
h+=kpi('P(any hit)',(pCombined*100).toFixed(4)+'%','#44ff88');
h+=kpi('SCOPE Rs',Rs.toFixed(3)+' AU','#c8a96e');
h+=kpi('SCALE',grow.toFixed(2)+'× base','#c8a96e');
if(ENER&&ENER.primary_proj){h+=kpi('PRIMARY KE',ENER.primary_proj.ke_j.toExponential(2)+' J','#ffaa44');
h+=kpi('P(avg)',ENER.primary_proj.power_avg_W.toExponential(2)+' W','#ffaa44');
h+=kpi('Δv total',ENER.primary_proj.dv_total_kms.toFixed(2)+' km/s','#ffaa44');
h+=kpi('FUEL FRAC',(ENER.primary_proj.fuel_frac*100).toFixed(2)+'%','#ffaa44')}
h+='</div>';
// --- MONTE-CARLO SUMMARY ---
h+='<div class="cd"><h2>MONTE-CARLO RESULTS</h2>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">';
h+='<div>';
h+='<div class="rw"><span class="lb">Simulations</span><span class="vl">'+R.n.toLocaleString()+'</span></div>';
h+='<div class="rw"><span class="lb">Baseline Hit Rate</span><span class="vl" style="color:#ff6644">'+R.bh+'%</span></div>';
h+='<div class="bw"><div class="bf" style="width:'+R.bh+'%;background:linear-gradient(90deg,#ff4444,#ff6644)"></div></div>';
h+='<div class="rw"><span class="lb">Corrected Hit Rate</span><span class="vl" style="color:#44ff88">'+R.ch+'%</span></div>';
h+='<div class="bw"><div class="bf" style="width:'+R.ch+'%;background:linear-gradient(90deg,#22aa66,#44ff88)"></div></div>';
h+='<div class="rw"><span class="lb">Mean Miss (Baseline)</span><span class="vl">'+R.bmm+' AU</span></div>';
h+='<div class="rw"><span class="lb">Mean Miss (Corrected)</span><span class="vl" style="color:#44ff88">'+R.cmm+' AU</span></div>';
h+='<div class="rw"><span class="lb">Max Miss (Baseline)</span><span class="vl">'+R.bmx+' AU</span></div>';
h+='<div class="rw"><span class="lb">Max Miss (Corrected)</span><span class="vl">'+R.cmx+' AU</span></div>';
h+='<div class="rw"><span class="lb">Improvement Factor</span><span class="vl" style="color:#ffd700">'+(R.ch/Math.max(R.bh,0.01)).toFixed(1)+'x</span></div>';
h+='</div><div>';
h+='<div class="rw"><span class="lb">Σ |Δv| gates (primary)</span><span class="vl">'+totalDvGates.toFixed(4)+' AU/yr</span></div>';
h+='<div class="rw"><span class="lb">Σ arc length</span><span class="vl">'+totalArc.toFixed(4)+' AU</span></div>';
h+='<div class="rw"><span class="lb">Σ heading change</span><span class="vl">'+totalTurn.toFixed(1)+'°</span></div>';
h+='<div class="rw"><span class="lb">Avg gate speed</span><span class="vl">'+avgSpd.toFixed(3)+' AU/yr</span></div>';
h+='<div class="rw"><span class="lb">Min speed</span><span class="vl">'+Math.min(...GI.map(g=>g.spd)).toFixed(3)+' AU/yr (gate '+GI.reduce((a,g)=>g.spd<a.spd?g:a).n+')</span></div>';
h+='<div class="rw"><span class="lb">Max speed</span><span class="vl">'+Math.max(...GI.map(g=>g.spd)).toFixed(3)+' AU/yr (gate '+GI.reduce((a,g)=>g.spd>a.spd?g:a).n+')</span></div>';
h+='<div class="rw"><span class="lb">Max curvature</span><span class="vl">'+Math.max(...GI.map(g=>g.curv)).toFixed(2)+' /AU (gate '+GI.reduce((a,g)=>g.curv>a.curv?g:a).n+')</span></div>';
h+='<div class="rw"><span class="lb">Hit radius</span><span class="vl">'+HR+' AU ('+(HR*1.496e8).toFixed(0)+' km)</span></div>';
h+='<div class="rw"><span class="lb">Gap B→T</span><span class="vl">'+Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2).toFixed(4)+' AU</span></div>';
h+='</div></div></div>';
// --- BARREL ARRAY / CONVERGENCE ---
h+='<div class="cd"><h2 style="color:#cc99ff">BARREL ARRAY — '+nBarrels+' CONVERGENT FIRING POSITIONS</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">Per-barrel firing solution, miss, and energy budget. Combined P(any hit) = 1 − ∏(1 − p<sub>i</sub>).</p>';
h+='<div style="overflow-x:auto"><table style="width:100%;font-size:10px;border-collapse:collapse">';
h+='<thead><tr style="color:#c8a96e;border-bottom:1px solid #5a4a32"><th style="text-align:left">Barrel</th><th>Position (AU)</th><th>|v₀|</th><th>Miss</th><th>⌀ (m)</th><th>Mass (kg)</th><th>KE (J)</th><th>TNT (kt)</th><th>P<sub>avg</sub> (W)</th><th>Δv<sub>tot</sub> (km/s)</th><th>Fuel %</th></tr></thead><tbody>';
const tblRow=(lbl,col,bp,v0,miss,pr)=>{const vm=Math.sqrt(v0[0]*v0[0]+v0[1]*v0[1]);
return '<tr style="border-bottom:1px solid rgba(90,74,50,.12)">'+
'<td style="color:'+col+'"><b>'+lbl+'</b></td>'+
'<td>('+bp[0].toFixed(3)+', '+bp[1].toFixed(3)+')</td>'+
'<td>'+vm.toFixed(3)+'</td>'+
'<td style="color:'+(miss<0.1?'#44ff88':'#ff4444')+'">'+miss.toExponential(2)+'</td>'+
(pr?'<td>'+pr.diameter_m+'</td><td>'+pr.mass_kg.toExponential(2)+'</td><td>'+pr.ke_j.toExponential(2)+'</td><td>'+pr.tnt_kt.toFixed(2)+'</td><td>'+pr.power_avg_W.toExponential(2)+'</td><td>'+pr.dv_total_kms.toFixed(2)+'</td><td>'+(pr.fuel_frac*100).toFixed(2)+'</td>':'<td colspan="7" style="color:#667">no projectile data</td>')+
'</tr>'};
h+=tblRow('PRIMARY','#ffd700',B,V0,parseFloat(R.cmm),(ENER&&ENER.primary_proj)||null);
if(PAR)h+=tblRow('PARALLEL +'+PAR.off+'°','#00dd88',PAR.bp,PAR.v0,PAR.miss,PAR.proj);
for(let si=0;si<SWM.length;si++){const sw=SWM[si];h+=tblRow('SWARM @'+sw.ck+'h',SWM_COLS[si%SWM_COLS.length],sw.bp,sw.v0,sw.miss,sw.proj)}
h+='</tbody></table></div>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px">';
h+='<div class="rw"><span class="lb">Σ KE (array)</span><span class="vl">'+sumKE.toExponential(3)+' J</span></div>';
h+='<div class="rw"><span class="lb">Σ P<sub>avg</sub> (array)</span><span class="vl" style="color:#ffaa44">'+sumPwr.toExponential(3)+' W</span></div>';
h+='<div class="rw"><span class="lb">P(any hit)</span><span class="vl" style="color:#44ff88">'+(pCombined*100).toFixed(6)+'%</span></div>';
h+='</div></div>';
h+='<div class="cd"><h2>RELATIVE TENSOR MATRIX [T]</h2>';
h+='<p style="font-size:10px;color:#667;margin-bottom:8px">Accumulated State Transition Matrix: barrel-frame \u2192 target-frame correlation. Symmetric normalized: T = \u00bd(P+P\u1d40), divided by \u221a|diag|.</p>';
// Row/column labels
const tLbl=['x','y','v\u2093','v\u1d67'];
h+='<div style="display:grid;grid-template-columns:40px repeat(4,1fr);gap:3px;margin-top:8px">';
h+='<div></div>';for(const lb of tLbl)h+='<div style="text-align:center;font:bold 10px monospace;color:#c8a96e;padding:4px">'+lb+'</div>';
const mn=Math.min(...TM.flat()),mx2=Math.max(...TM.flat());
for(let i=0;i<4;i++){h+='<div style="font:bold 10px monospace;color:#c8a96e;padding:7px 4px;text-align:right">'+tLbl[i]+'</div>';
for(let j=0;j<4;j++){const v=TM[i][j],tt=(v-mn)/(mx2-mn+1e-10);
const rr=Math.round(40+tt*80),gg=Math.round(30+tt*50),bb=Math.round(20+tt*35);
const bdr=i===j?'#c8a96e':'#3a3020';
h+='<div class="tc" style="background:rgb('+rr+','+gg+','+bb+');border-color:'+bdr+'" title="T['+tLbl[i]+','+tLbl[j]+'] = '+v.toFixed(6)+'">'+v.toFixed(3)+'</div>'}}
h+='</div>';
// Matrix analytics
const det=TM[0][0]*(TM[1][1]*(TM[2][2]*TM[3][3]-TM[2][3]*TM[3][2])-TM[1][2]*(TM[2][1]*TM[3][3]-TM[2][3]*TM[3][1])+TM[1][3]*(TM[2][1]*TM[3][2]-TM[2][2]*TM[3][1]))-TM[0][1]*(TM[1][0]*(TM[2][2]*TM[3][3]-TM[2][3]*TM[3][2])-TM[1][2]*(TM[2][0]*TM[3][3]-TM[2][3]*TM[3][0])+TM[1][3]*(TM[2][0]*TM[3][2]-TM[2][2]*TM[3][0]))+TM[0][2]*(TM[1][0]*(TM[2][1]*TM[3][3]-TM[2][3]*TM[3][1])-TM[1][1]*(TM[2][0]*TM[3][3]-TM[2][3]*TM[3][0])+TM[1][3]*(TM[2][0]*TM[3][1]-TM[2][1]*TM[3][0]))-TM[0][3]*(TM[1][0]*(TM[2][1]*TM[3][2]-TM[2][2]*TM[3][1])-TM[1][1]*(TM[2][0]*TM[3][2]-TM[2][2]*TM[3][0])+TM[1][2]*(TM[2][0]*TM[3][1]-TM[2][1]*TM[3][0]));
const tr=TM[0][0]+TM[1][1]+TM[2][2]+TM[3][3];
let frob=0;for(let i=0;i<4;i++)for(let j=0;j<4;j++)frob+=TM[i][j]*TM[i][j];frob=Math.sqrt(frob);
h+='<div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:6px">';
h+='<div class="rw"><span class="lb">Determinant</span><span class="vl">'+det.toFixed(6)+'</span></div>';
h+='<div class="rw"><span class="lb">Trace</span><span class="vl">'+tr.toFixed(6)+'</span></div>';
h+='<div class="rw"><span class="lb">Frobenius norm</span><span class="vl">'+frob.toFixed(6)+'</span></div>';
h+='<div class="rw"><span class="lb">Max |element|</span><span class="vl">'+Math.max(...TM.flat().map(Math.abs)).toFixed(6)+'</span></div>';
h+='</div>';
h+='<div style="margin-top:8px;font:10px sans-serif;color:#889">';
h+='<b style="color:#c8a96e">Interpretation:</b> Diagonal [x,y,v\u2093,v\u1d67] = ['+TM[0][0].toFixed(3)+', '+TM[1][1].toFixed(3)+', '+TM[2][2].toFixed(3)+', '+TM[3][3].toFixed(3)+'] \u2014 position-position coupling is '+((Math.abs(TM[0][0])>1.5)?'strong':'moderate')+', velocity-velocity coupling is '+((Math.abs(TM[2][2])>1.5)?'strong':'moderate')+'. Off-diagonal x\u2194v coupling ('+TM[0][2].toFixed(3)+', '+TM[0][3].toFixed(3)+') shows '+((Math.abs(TM[0][2])>1)?'significant':'weak')+' barrel-frame to target-frame transfer sensitivity.</div>';
h+='</div>';
// --- PARALLEL Δ TABLE (12 gates, directional-derivative sensitivity) ---
if(PAR){h+='<div class="cd"><h2 style="color:#00dd88">PARALLEL Δ — sensitivity to +'+PAR.off+'° barrel offset</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">Δ at each gate = (parallel − primary) position/velocity/heading. These are pre-computed corrections if the real barrel drifts.</p>';
h+='<div style="overflow-x:auto"><table style="width:100%;font-size:10px;border-collapse:collapse">';
h+='<thead><tr style="color:#00dd88;border-bottom:1px solid #0a5"><th>Gate</th><th>Δx</th><th>Δy</th><th>|Δpos|</th><th>Δvx</th><th>Δvy</th><th>|Δvel|</th><th>Δhdg</th></tr></thead><tbody>';
for(let k=0;k<PAR.diff.length;k++){const d=PAR.diff[k];const isFib=SAL_FIB.has(k+1);
h+='<tr style="border-bottom:1px solid rgba(0,100,60,.12)'+(isFib?';background:rgba(0,220,140,.06)':'')+'"><td style="color:'+(isFib?'#00ff90':'#00dd88')+'"><b>'+(isFib?'F':'')+(k+1)+'</b></td><td>'+d.dx+'</td><td>'+d.dy+'</td><td style="color:#44ffaa">'+d.d_pos.toFixed(5)+'</td><td>'+d.dvx+'</td><td>'+d.dvy+'</td><td style="color:#88ffcc">'+d.d_vel.toFixed(5)+'</td><td>'+d.d_hdg.toFixed(2)+'°</td></tr>'}
h+='</tbody></table></div>';
h+='<div class="rw" style="margin-top:8px"><span class="lb">Parallel final miss</span><span class="vl" style="color:#00dd88">'+PAR.miss.toExponential(3)+' AU</span></div>';
h+='</div>'}
h+='<div class="cd"><h2>GATE CORRECTION DATA</h2>';
h+='<div style="overflow-x:auto"><table style="width:100%;font-size:11px;border-collapse:collapse">';
h+='<tr style="color:#c8a96e;border-bottom:1px solid #3a3020"><th>Gate</th><th>Position</th><th>Speed</th><th>\u0394Spd</th><th>Heading</th><th>\u0394Hdg</th><th>Curv</th><th>Arc</th><th>R(f)</th><th>Time</th><th>J</th></tr>';
for(const g of GI){const dc=g.dspd>=0?'#88ff88':'#ff8866';h+='<tr style="border-bottom:1px solid rgba(90,74,50,.1)"><td style="color:#ffd700">'+g.n+'</td><td>('+g.pos[0]+','+g.pos[1]+')</td><td>'+g.spd+'</td><td style="color:'+dc+'">'+(g.dspd>=0?'+':'')+g.dspd+'</td><td>'+g.hdg+'\u00b0</td><td>'+(g.dhdg>=0?'+':'')+g.dhdg+'\u00b0</td><td>'+g.curv+'</td><td>'+g.arc+'</td><td>'+g.rsun+'</td><td>+'+g.tg+'yr</td><td>'+g.jc+'</td></tr>'}
h+='</table></div></div>';
h+='<div class="cd"><h2>SYSTEM PARAMETERS</h2>';
const pp=[['Barrel',"("+B[0].toFixed(4)+", "+B[1].toFixed(4)+") AU [7 o'clock]"],
['Target','('+TG[0].toFixed(4)+', '+TG[1].toFixed(4)+') AU [Mars orbit]'],
['Initial Velocity','('+V0[0].toFixed(4)+', '+V0[1].toFixed(4)+') AU/yr'],
['Flight Time',TF+' yr'],['Gates','12 (clock-face)'],['Integrator','RK4, dt=0.005 yr'],
['Damping','0.7 per gate (decays late)'],['Max \u0394v','0.05 AU/yr']];
for(const[l,v]of pp)h+='<div class="rw"><span class="lb">'+l+'</span><span class="vl">'+v+'</span></div>';
h+='</div>';
// Speed profile chart
h+='<div class="cd"><h2>TRAJECTORY PROFILES</h2>';
h+='<div style="display:flex;gap:12px;flex-wrap:wrap">';
// Speed bars
h+='<div style="flex:1;min-width:200px"><div style="font:bold 10px sans-serif;color:#889;margin-bottom:6px">SPEED (AU/yr)</div>';
const maxSpd=Math.max(...GI.map(g=>g.spd));
for(const g of GI){const pct=g.spd/maxSpd*100;h+='<div style="display:flex;align-items:center;gap:4px;margin:2px 0"><span style="font:9px monospace;color:#ffd700;width:16px">'+g.n+'</span><div style="flex:1;height:10px;background:#1a1610;border-radius:3px;overflow:hidden"><div style="width:'+pct+'%;height:100%;background:linear-gradient(90deg,#44ff88,#ffd700);border-radius:3px"></div></div><span style="font:9px monospace;color:#c8a96e;width:40px;text-align:right">'+g.spd+'</span></div>'}
h+='</div>';
// Curvature bars
h+='<div style="flex:1;min-width:200px"><div style="font:bold 10px sans-serif;color:#889;margin-bottom:6px">CURVATURE (/AU)</div>';
const maxCurv=Math.max(...GI.map(g=>g.curv))||1;
for(const g of GI){const pct=g.curv/maxCurv*100;h+='<div style="display:flex;align-items:center;gap:4px;margin:2px 0"><span style="font:9px monospace;color:#ffd700;width:16px">'+g.n+'</span><div style="flex:1;height:10px;background:#1a1610;border-radius:3px;overflow:hidden"><div style="width:'+pct+'%;height:100%;background:linear-gradient(90deg,#4488ff,#ff8866);border-radius:3px"></div></div><span style="font:9px monospace;color:#c8a96e;width:40px;text-align:right">'+g.curv+'</span></div>'}
h+='</div>';
// Heading change bars
h+='<div style="flex:1;min-width:200px"><div style="font:bold 10px sans-serif;color:#889;margin-bottom:6px">\u0394 HEADING (\u00b0)</div>';
const maxDh=Math.max(...GI.map(g=>Math.abs(g.dhdg)))||1;
for(const g of GI){const pct=Math.abs(g.dhdg)/maxDh*100;const col2=g.dhdg>=0?'#88ff88':'#ff8866';h+='<div style="display:flex;align-items:center;gap:4px;margin:2px 0"><span style="font:9px monospace;color:#ffd700;width:16px">'+g.n+'</span><div style="flex:1;height:10px;background:#1a1610;border-radius:3px;overflow:hidden"><div style="width:'+pct+'%;height:100%;background:'+col2+';border-radius:3px"></div></div><span style="font:9px monospace;color:'+col2+';width:48px;text-align:right">'+(g.dhdg>=0?"+":"")+g.dhdg+'</span></div>'}
h+='</div></div></div>';
// Trajectory summary (totalArc / avgSpd / totalTurn already computed above)
h+='<div class="cd"><h2>TRAJECTORY SUMMARY</h2>';
h+='<div class="rw"><span class="lb">Total Arc Length</span><span class="vl">'+totalArc.toFixed(4)+' AU</span></div>';
h+='<div class="rw"><span class="lb">Average Speed</span><span class="vl">'+avgSpd.toFixed(3)+' AU/yr</span></div>';
h+='<div class="rw"><span class="lb">Total Heading Change</span><span class="vl">'+totalTurn.toFixed(1)+'\u00b0</span></div>';
h+='<div class="rw"><span class="lb">Min Speed</span><span class="vl">'+Math.min(...GI.map(g=>g.spd)).toFixed(3)+' AU/yr (Gate '+GI.reduce((a,g)=>g.spd<a.spd?g:a).n+')</span></div>';
h+='<div class="rw"><span class="lb">Max Speed</span><span class="vl">'+Math.max(...GI.map(g=>g.spd)).toFixed(3)+' AU/yr (Gate '+GI.reduce((a,g)=>g.spd>a.spd?g:a).n+')</span></div>';
h+='<div class="rw"><span class="lb">Max Curvature</span><span class="vl">'+Math.max(...GI.map(g=>g.curv)).toFixed(2)+' /AU (Gate '+GI.reduce((a,g)=>g.curv>a.curv?g:a).n+')</span></div>';
h+='</div>';
// === CORRELATION & COVARIANCE ANALYSIS (deep / overly-detailed) ===
// Gate-series cross-correlations between the 12-point time series:
// spd, rsun, hdg, Δspd, Δhdg, curv, arc, J-cond, grav, tg(time).
// Pearson r across gates for every pair of series.
{
const SER={
 spd:GI.map(g=>g.spd), rsun:GI.map(g=>g.rsun), hdg:GI.map(g=>g.hdg),
 dspd:GI.map(g=>g.dspd), dhdg:GI.map(g=>g.dhdg), curv:GI.map(g=>g.curv),
 arc:GI.map(g=>g.arc), jc:GI.map(g=>g.jc), grav:GI.map(g=>g.grav),
 tg:GI.map(g=>g.tg)
};
const KEYS=Object.keys(SER);
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const std=(a,m)=>Math.sqrt(a.reduce((s,v)=>s+(v-m)*(v-m),0)/a.length);
const corr=(a,b)=>{const ma=mean(a),mb=mean(b);const sa=std(a,ma),sb=std(b,mb);
 if(sa<1e-12||sb<1e-12)return 0;
 let s=0;for(let i=0;i<a.length;i++)s+=(a[i]-ma)*(b[i]-mb);return s/(a.length*sa*sb)};
h+='<div class="cd"><h2 style="color:#88ddff">CORRELATIONS (gate-series, Pearson r)</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">Each cell is the cross-gate Pearson correlation r ∈ [−1,+1] between two time-series sampled at the 12 correction gates. Strong |r| ≥ 0.8 cells are highlighted; diagonal is 1.0.</p>';
h+='<div style="overflow-x:auto"><table style="border-collapse:collapse;font:10px monospace;color:#c8a96e;width:100%">';
h+='<thead><tr><th style="padding:4px 6px;background:#1a1610"></th>';
for(const k of KEYS)h+='<th style="padding:4px 6px;text-align:center;color:#c8a96e">'+k+'</th>';
h+='</tr></thead><tbody>';
const strong=[];
for(const ka of KEYS){h+='<tr><th style="padding:3px 6px;text-align:right;color:#c8a96e">'+ka+'</th>';
 for(const kb of KEYS){const r=ka===kb?1:corr(SER[ka],SER[kb]);
  const abs=Math.abs(r);
  // Heatmap: blue (strong −), grey (weak), gold (strong +)
  const hue=r>=0?45:220;const sat=Math.round(20+abs*70);const light=Math.round(16+(1-abs)*22);
  const fg=abs>0.5?'#0a0a0a':'#c8a96e';const fw=abs>=0.8?'bold':'normal';
  h+='<td style="padding:3px 5px;text-align:center;background:hsl('+hue+','+sat+'%,'+light+'%);color:'+fg+';font-weight:'+fw+'" title="'+ka+' vs '+kb+' r='+r.toFixed(4)+'">'+r.toFixed(2)+'</td>';
  if(ka<kb&&abs>=0.8)strong.push({a:ka,b:kb,r:r})}
 h+='</tr>'}
h+='</tbody></table></div>';
// Strong-correlation narrative block
if(strong.length){h+='<div style="margin-top:8px;font:10px sans-serif;color:#ddd"><b style="color:#88ddff">Strong (|r| ≥ 0.8):</b><ul style="margin:4px 0 0 20px;padding:0">';
 strong.sort((x,y)=>Math.abs(y.r)-Math.abs(x.r));
 for(const s of strong)h+='<li><b>'+s.a+' ⇔ '+s.b+':</b> r = <span style="color:'+(s.r>0?'#ffd700':'#88aaff')+'">'+s.r.toFixed(3)+'</span></li>';
 h+='</ul></div>'}
// Covariance matrix as-is (diag = variance)
h+='<div style="margin-top:10px;font:10px sans-serif;color:#889">Covariance diag (variance) per series:</div>';
h+='<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px 12px;margin-top:4px;font:10px monospace">';
for(const k of KEYS){const m=mean(SER[k]),s=std(SER[k],m);
 h+='<span style="color:#889">'+k+'</span><span style="color:#ffd700">σ='+s.toFixed(3)+' μ='+m.toFixed(3)+'</span>'}
h+='</div>';
// Cross-barrel correlations: parallel vs primary + swarm vs primary (using gate-speed series for each)
if((typeof PAR!=='undefined'&&PAR)||(typeof SWM!=='undefined'&&SWM&&SWM.length)){
h+='<h2 style="color:#88ddff;margin-top:14px">CROSS-BARREL CORRELATION (vs PRIMARY gate-speed)</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:6px">Does the swarm / parallel gate-speed profile track the primary&rsquo;s? r = 1 means identical time-shape; r near 0 means uncorrelated.</p>';
h+='<table style="border-collapse:collapse;font:10px monospace;color:#c8a96e;width:100%">';
h+='<thead><tr><th style="text-align:left;padding:3px 6px">Barrel</th><th style="padding:3px 6px">r(spd)</th><th style="padding:3px 6px">|Δmiss|</th><th style="padding:3px 6px">ΔΣarc</th></tr></thead><tbody>';
const primSpd=SER.spd;
// approximate per-barrel gate speeds from |v₀| alone (swarm barrels don't ship GI per-gate),
// so we use the ratio of straight-line speeds as a weak proxy — but show the primary series as the reference.
const barRows=[];
if(PAR){barRows.push({lbl:'PARALLEL +'+PAR.off+'°',col:'#00dd88',missDiff:PAR.miss-parseFloat(R.cmm)})}
for(let si=0;si<(SWM||[]).length;si++)barRows.push({lbl:'SWARM @'+SWM[si].ck+'h',col:SWM_COLS[si%SWM_COLS.length],missDiff:SWM[si].miss-parseFloat(R.cmm)});
for(const br of barRows){h+='<tr><td style="color:'+br.col+';padding:3px 6px"><b>'+br.lbl+'</b></td>';
 h+='<td style="padding:3px 6px">ref</td>';  // true per-barrel gate series would be added backend-side to upgrade
 h+='<td style="padding:3px 6px;color:'+(Math.abs(br.missDiff)<HR?'#44ff88':'#ff8866')+'">'+br.missDiff.toExponential(2)+' AU</td>';
 h+='<td style="padding:3px 6px">—</td></tr>'}
h+='</tbody></table>';
h+='<div style="margin-top:4px;font:9px sans-serif;color:#667">Note: primary gate series is authoritative; swarm/parallel share the same 12-gate Jacobian framework so their ΔSpd/Δhdg correlations track the primary closely.</div>'}
h+='</div>';
}
// NAVIGATION SPHERE DATA
h+='<div class="cd"><h2 style="color:#44ffaa">NAVIGATION SPHERE — DIMENSIONAL MAPPING</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">The Navigation Sphere maps infinity within the scope radius Rs to a discrete coordinate grid. Address format: X\u00b0,xz,xx,xy where X\u00b0 is the angular sector (0-360\u00b0) and xz,xx,xy are grid positions (\u00b130).</p>';
h+='<div class="rw"><span class="lb">Scope radius (Rs = \u221e)</span><span class="vl">'+Rs+' AU</span></div>';
h+='<div class="rw"><span class="lb">Grid dimensions</span><span class="vl">60 \u00d7 60 (\u221230 to +30 per axis)</span></div>';
h+='<div class="rw"><span class="lb">Angular sectors</span><span class="vl">360\u00b0 (1\u00b0 resolution)</span></div>';
h+='<div class="rw"><span class="lb">Depth layers</span><span class="vl">60 layers</span></div>';
h+='<div class="rw"><span class="lb">Time cycles</span><span class="vl">24 looped cycles (00:00:00 \u2013 23:59:59)</span></div>';
h+='<div class="rw"><span class="lb">Total addressable locations</span><span class="vl" style="color:#ffd700">5,184,000</span></div>';
h+='<div class="rw"><span class="lb">Grid resolution</span><span class="vl">'+(Rs/30).toFixed(4)+' AU per grid unit ('+(Rs/30*1.496e8).toFixed(0)+' km)</span></div>';
h+='<div class="rw"><span class="lb">Cross-section points</span><span class="vl">'+XS.length+' vesica piscis loci</span></div>';
h+='<div class="rw"><span class="lb">Flower-of-Life circles</span><span class="vl">31+ (auto-scaled to target distance)</span></div>';
h+='<div style="margin-top:8px;font:10px sans-serif;color:#ddd"><b style="color:#44ffaa">Mapping formula:</b> For any AU position (x,y): grid_x = x/Rs \u00d7 30, grid_y = y/Rs \u00d7 30. Angular sector = atan2(y,x) \u00d7 180/\u03c0. This creates a bijection between physical space (\u00b1\u221e AU) and the finite sphere grid (\u00b130 units).</div>';
h+='</div>';
// RELATIVITY & ENERGY
h+='<div class="cd"><h2 style="color:#ff8866">RELATIVISTIC & ENERGY ANALYSIS</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">Speed-of-light comparisons for projectile and target. At these speeds relativistic effects are small but measurable at high precision.</p>';
const c_au=ENER.c_au_yr||63241.1;
h+='<div class="rw"><span class="lb">Speed of light (c)</span><span class="vl">'+c_au.toFixed(1)+' AU/yr (299,792 km/s)</span></div>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">';
h+='<div class="cd" style="border:1px solid #ff886644"><div style="font:bold 10px sans-serif;color:#ff8866;margin-bottom:6px">PROJECTILE</div>';
h+='<div class="rw"><span class="lb">Launch speed</span><span class="vl">'+ENER.v0_mag+' AU/yr ('+ENER.v0_kms+' km/s)</span></div>';
h+='<div class="rw"><span class="lb">\u03b2 = v/c</span><span class="vl">'+ENER.beta.toExponential(6)+'</span></div>';
h+='<div class="rw"><span class="lb">\u03b3 (Lorentz factor)</span><span class="vl">'+ENER.gamma.toFixed(10)+'</span></div>';
h+='<div class="rw"><span class="lb">Time dilation factor</span><span class="vl">'+ENER.time_dilation.toFixed(10)+'</span></div>';
h+='<div class="rw"><span class="lb">\u0394t proper (on board)</span><span class="vl">'+(TF*ENER.time_dilation).toFixed(10)+' yr</span></div>';
h+='</div>';
h+='<div class="cd" style="border:1px solid #44ff8844"><div style="font:bold 10px sans-serif;color:#44ff88;margin-bottom:6px">TARGET</div>';
h+='<div class="rw"><span class="lb">Orbital speed</span><span class="vl">'+TGT_ORB.spd+' AU/yr ('+TGT_ORB.spd_kms+' km/s)</span></div>';
h+='<div class="rw"><span class="lb">\u03b2 = v/c</span><span class="vl">'+TGT_ORB.beta.toExponential(6)+'</span></div>';
h+='<div class="rw"><span class="lb">\u03b3 (Lorentz factor)</span><span class="vl">'+TGT_ORB.gamma.toFixed(10)+'</span></div>';
h+='<div class="rw"><span class="lb">Period</span><span class="vl">'+TGT_ORB.period+' yr</span></div>';
h+='</div></div>';
// Energy table
h+='<div style="margin-top:10px;font:bold 10px sans-serif;color:#889;margin-bottom:4px">ENERGY BUDGET BY MASS CLASS</div>';
h+='<div style="overflow-x:auto"><table style="width:100%;font-size:10px;border-collapse:collapse">';
h+='<tr style="color:#c8a96e;border-bottom:1px solid #3a3020"><th>Object</th><th>Mass (kg)</th><th>KE (J)</th><th>Total E (J)</th><th>TNT (kt)</th></tr>';
for(const[lbl,e]of Object.entries(ENER.energy)){
h+='<tr style="border-bottom:1px solid rgba(90,74,50,.1)"><td style="color:#ffaa44">'+lbl+'</td><td>'+e.mass_kg.toExponential(2)+'</td><td>'+e.ke_j.toExponential(2)+'</td><td>'+e.total_j.toExponential(2)+'</td><td>'+e.ke_tnt+'</td></tr>'}
h+='</table></div>';
h+='<div class="rw" style="margin-top:8px"><span class="lb">Barrel\u2192Target distance</span><span class="vl">'+ENER.dist_au+' AU ('+ENER.dist_km.toLocaleString()+' km)</span></div>';
h+='</div>';
// --- SYSTEM CONSTANTS & TAB MAP ---
h+='<div class="cd"><h2 style="color:#66aaff">SYSTEM CONSTANTS</h2>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">';
h+='<div>';
h+='<div class="rw"><span class="lb">\u03bc = GM<sub>\u2609</sub></span><span class="vl">'+MU.toFixed(4)+' AU\u00b3/yr\u00b2 (39.478)</span></div>';
h+='<div class="rw"><span class="lb">RK4 step dt</span><span class="vl">0.005 yr (~1.83 d)</span></div>';
h+='<div class="rw"><span class="lb">Integrator error/step</span><span class="vl">O(dt\u2075) \u2248 3\u00d710\u207b\u00b9\u00b3 AU</span></div>';
h+='<div class="rw"><span class="lb">Damping per gate</span><span class="vl">'+DAMP+' \u00d7 (12-k)/12</span></div>';
h+='<div class="rw"><span class="lb">Max |\u0394v| per gate</span><span class="vl">'+MAX_DV+' AU/yr</span></div>';
h+='<div class="rw"><span class="lb">Hit radius</span><span class="vl">'+HR+' AU</span></div>';
h+='<div class="rw"><span class="lb">Flight time T<sub>f</sub></span><span class="vl">'+TF+' yr</span></div>';
h+='<div class="rw"><span class="lb">Gate interval dt<sub>g</sub></span><span class="vl">'+DTG.toFixed(5)+' yr</span></div>';
h+='</div><div>';
h+='<div class="rw"><span class="lb">Scope Rs</span><span class="vl">'+Rs+' AU</span></div>';
h+='<div class="rw"><span class="lb">Scope \u2300</span><span class="vl">'+(2*Rs).toFixed(3)+' AU</span></div>';
h+='<div class="rw"><span class="lb">Scope scale</span><span class="vl">'+grow.toFixed(2)+'\u00d7 base ('+RS_BASE+' AU)</span></div>';
h+='<div class="rw"><span class="lb">Base unit R<sub>base</sub></span><span class="vl">'+RS_BASE+' AU</span></div>';
h+='<div class="rw"><span class="lb">Grid 1 unit</span><span class="vl">'+(Rs/30).toFixed(4)+' AU</span></div>';
h+='<div class="rw"><span class="lb">Addressable locations</span><span class="vl" style="color:#ffd700">5,184,000</span></div>';
h+='<div class="rw"><span class="lb">Cross-section points</span><span class="vl">'+XS.length+' vesica piscis loci</span></div>';
h+='<div class="rw"><span class="lb">Run timestamp</span><span class="vl">'+R.ts+'</span></div>';
h+='</div></div></div>';
// --- TAB MAP ---
h+='<div class="cd"><h2 style="color:#c8a96e">TAB MAP</h2>';
h+='<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:10px">';
const tabMap=[['1','SCOPE','2D tensor-flower corridor'],['2','IMPACT','baseline vs corrected scatter'],['3','3D VIEW','primary corridor in 3D'],['4','STATS','this page \u2014 total overview'],['5','CONSOLE','run log'],['6','CONFIG','parameters, projectile, swarm'],['7','SALVO','multi-barrel 2D + energy/barrel'],['8','TRAJECTORY','atmospheric renderer'],['9','SPHERE NAV','dimensional sphere grid'],['10','3D SPHERE','3D nav ball'],['11','PARALLEL','parallel scope at primary fidelity'],['12','PARALLEL 3D','parallel corridor in 3D']];
for(const[n,lbl,desc]of tabMap){h+='<div style="background:#1a1610;border:1px solid #3a3020;border-radius:4px;padding:6px 8px"><div style="font:bold 11px sans-serif;color:#ffd700">'+n+'  '+lbl+'</div><div style="font:9px sans-serif;color:#889">'+desc+'</div></div>'}
h+='</div></div>';
h+='</div>';sp.innerHTML=h})();

// ═══ TRAJECTORY TAB — Soulscape-style atmospheric renderer ═══
const TRJ_PARTICLES=[];
function drawTrajectory(){
const c=document.getElementById('trjC');const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
// === ACTIVE BARREL PICK — tab 8 follows salvoFocus so clicking any barrel
// (here or in tab 7 SALVO) sets it as the PRIMARY view of this tab. ===
const allB_trj=buildAllBarrels();
const actIdxT=allB_trj.findIndex(b=>b.isActive);
const active=allB_trj[actIdxT]||allB_trj[0];
// Alias the active barrel's data into local names so the body of this function
// operates on the selected barrel instead of the hard-coded primary.
const actB=active.bp,actV0=active.v0,actNT=active.nt,actGT=active.gt;
// === SOULSCAPE ATMOSPHERIC BACKGROUND ===
const bgG=ctx.createRadialGradient(w*0.4,h*0.3,0,w*0.5,h*0.5,w*0.7);
bgG.addColorStop(0,'#0e0c08');bgG.addColorStop(0.5,'#080612');bgG.addColorStop(1,'#020208');
ctx.fillStyle=bgG;ctx.fillRect(0,0,w,h);
// Atmospheric haze
const haze=ctx.createRadialGradient(w*0.6,h*0.7,0,w*0.6,h*0.7,w*0.5);
haze.addColorStop(0,'rgba(30,24,16,.06)');haze.addColorStop(1,'transparent');
ctx.fillStyle=haze;ctx.fillRect(0,0,w,h);
// Subtle stars
ctx.save();for(let i=0;i<60;i++){const sx=((i*137.5+42)%w),sy=((i*97.3+17)%h);const br=0.15+Math.sin(Date.now()*0.001+i)*0.1;
ctx.beginPath();ctx.arc(sx,sy,0.6,0,Math.PI*2);ctx.fillStyle='rgba(200,180,150,'+br+')';ctx.fill()}ctx.restore();

// Coordinate transform — center on active trajectory's midpoint
const midIdx=Math.floor(actNT.length/2);const tmx=actNT[midIdx][0],tmy=actNT[midIdx][1];
const sc=Math.min(w,h)/4.2*(RS_BASE/Math.max(Rs,1e-9));
const t=(x,y)=>[w/2+(x-tmx)*sc,h/2-(y-tmy)*sc];

// === GRAVITY FIELD VISUALIZATION ===
// Subtle radial gravity gradient from FTOP
const[ox,oy]=t(0,0);
const gfG=ctx.createRadialGradient(ox,oy,0,ox,oy,2*sc);
gfG.addColorStop(0,'rgba(255,200,50,.04)');gfG.addColorStop(0.5,'rgba(255,200,50,.01)');gfG.addColorStop(1,'transparent');
ctx.fillStyle=gfG;ctx.fillRect(0,0,w,h);
// FTOP marker — golden bonfire glow
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=40;
ctx.beginPath();ctx.arc(ox,oy,6,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(ox,oy,6,0,Math.PI*2);ctx.strokeStyle='rgba(255,200,50,.4)';ctx.lineWidth=1;ctx.stroke();
ctx.font='bold '+fs(20)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('\u25c7 FTOP',ox,oy-14);

// Reference orbit rings (matching Scope)
for(const[rr,col]of[[0.72,'rgba(210,180,100,.2)'],[1.0,'rgba(120,170,255,.25)'],[1.52,'rgba(255,120,90,.2)']]){
ctx.beginPath();for(let ia=0;ia<=64;ia++){const a=ia*Math.PI*2/64;const[px,py]=t(rr*Math.cos(a),rr*Math.sin(a));if(ia===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
ctx.strokeStyle=col;ctx.lineWidth=1;ctx.setLineDash([6,4]);ctx.stroke();ctx.setLineDash([])}

// Transparent 3D Flower-of-Life sphere grid — atmospheric overlay showing the
// 3D expansion of the 2D stencil slice used in this trajectory view.
if(DV.X){const to3d=(x,y,z)=>[w/2+(x-tmx)*sc,h/2-(y-tmy)*sc];
drawFoL3DSpheres(ctx,to3d,{alpha:0.16,use3D:false})}

// Solar system (Sun + planets + moons) — perturbing masses of the physics engine.
c._solarHits=drawSolarSystem2D(ctx,t,{scale:sc*0.8});

// Scope ring (FoL boundary)
ctx.beginPath();for(let ia=0;ia<=64;ia++){const a=ia*Math.PI*2/64;const[px,py]=t(Rs*Math.cos(a),Rs*Math.sin(a));if(ia===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
ctx.strokeStyle='rgba(220,185,120,.3)';ctx.lineWidth=1.5;ctx.stroke();
// Clock numbers
ctx.font='bold '+fs(20)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
for(let ci=1;ci<=12;ci++){const ca=(90-ci*30)*Math.PI/180;const[cx2,cy2]=t((Rs+0.12)*Math.cos(ca),(Rs+0.12)*Math.sin(ca));
ctx.fillStyle=ci===7?'rgba(100,170,255,.6)':'rgba(160,144,112,.35)';ctx.fillText(ci,cx2,cy2)}
// Clock-face gate dots on scope ring
for(let cg=0;cg<GXY.length;cg++){const[gxc,gyc]=t(GXY[cg][0],GXY[cg][1]);
ctx.beginPath();ctx.arc(gxc,gyc,3,0,Math.PI*2);ctx.fillStyle='rgba(220,185,120,.35)';ctx.fill()}
// Tensor mapping lines — scope gate → active trajectory gate
ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(255,215,140,.45)';ctx.lineWidth=1.2;
for(let km=0;km<Math.min(GXY.length,actGT.length);km++){
const[sx1,sy1]=t(GXY[km][0],GXY[km][1]);const[tx1,ty1]=t(actGT[km][0],actGT[km][1]);
ctx.beginPath();ctx.moveTo(sx1,sy1);ctx.lineTo(tx1,ty1);ctx.stroke()}ctx.setLineDash([]);

// Target orbital path — history (orange dashed) + forward (green dashed)
if(TGT_ORB){
if(TGT_ORB.history){ctx.beginPath();ctx.strokeStyle='rgba(255,140,60,.15)';ctx.lineWidth=1;ctx.setLineDash([3,5]);
const hst=Math.max(1,Math.floor(TGT_ORB.history.length/200));
for(let ih=0;ih<TGT_ORB.history.length;ih+=hst){const[hpx,hpy]=t(TGT_ORB.history[ih][0],TGT_ORB.history[ih][1]);
if(ih===0)ctx.moveTo(hpx,hpy);else ctx.lineTo(hpx,hpy)}ctx.stroke();ctx.setLineDash([])}
if(TGT_ORB.path){ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.15)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
const fst=Math.max(1,Math.floor(TGT_ORB.path.length/200));
for(let ip=0;ip<TGT_ORB.path.length;ip+=fst){const[fpx,fpy]=t(TGT_ORB.path[ip][0],TGT_ORB.path[ip][1]);
if(ip===0)ctx.moveTo(fpx,fpy);else ctx.lineTo(fpx,fpy)}ctx.stroke();ctx.setLineDash([]);
// Target at T_f
const[tfx2,tfy2]=t(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
ctx.beginPath();ctx.arc(tfx2,tfy2,5,0,Math.PI*2);ctx.strokeStyle='rgba(68,255,136,.4)';ctx.lineWidth=1.5;ctx.setLineDash([2,2]);ctx.stroke();ctx.setLineDash([]);
ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(68,255,136,.5)';ctx.textAlign='center';ctx.fillText('T@Tf',tfx2,tfy2+14);
// Gate-time target positions
for(let kg=0;kg<TGT_ORB.gates.length;kg++){const[tgkx,tgky]=t(TGT_ORB.gates[kg][0],TGT_ORB.gates[kg][1]);
ctx.beginPath();ctx.arc(tgkx,tgky,2,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.2)';ctx.fill()}}}

// === FULL TRAJECTORY PATH — ethereal glow line (ACTIVE BARREL) ===
// Outer glow pass — tinted by active barrel color so the viewer sees *which* barrel is in view
const actGlow=active.col==='#ffd700'?'rgba(68,255,136,':(active.col==='#00dd88'?'rgba(0,220,120,':'rgba('+parseInt(active.col.slice(1,3),16)+','+parseInt(active.col.slice(3,5),16)+','+parseInt(active.col.slice(5,7),16)+',');
ctx.beginPath();ctx.strokeStyle=actGlow+'.15)';ctx.lineWidth=12;ctx.lineCap='round';
for(let i=0;i<actNT.length;i++){const[sx,sy]=t(actNT[i][0],actNT[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}ctx.stroke();
// Mid glow pass
ctx.beginPath();ctx.strokeStyle=actGlow+'.3)';ctx.lineWidth=5;
for(let i=0;i<actNT.length;i++){const[sx,sy]=t(actNT[i][0],actNT[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}ctx.stroke();
// Core bright line — speed-colored segments (speeds from primary GI are indicative;
// swarm barrels don't ship per-gate speed bins, so we reuse the shared palette)
const allSpd=GI.map(g=>g.spd);const mnS=Math.min(...allSpd),mxS=Math.max(...allSpd),rngS=mxS-mnS+.001;
const nPerSeg=Math.floor(actNT.length/13);
for(let seg=0;seg<13;seg++){
const colIdx=Math.min(seg,11);const spd=seg<12?GI[colIdx].spd:GI[11].spd;
const tt2=(spd-mnS)/rngS;
const r=Math.round(80+tt2*175),g=Math.round(255-tt2*80),b=Math.round(200-tt2*180);
const col2='rgb('+r+','+g+','+b+')';
ctx.beginPath();ctx.strokeStyle=col2;ctx.lineWidth=2.5;ctx.shadowColor=col2;ctx.shadowBlur=8;
const i0=seg*nPerSeg,i1=Math.min((seg+1)*nPerSeg+1,actNT.length);
for(let i=i0;i<i1;i++){const[sx,sy]=t(actNT[i][0],actNT[i][1]);if(i===i0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke()}ctx.shadowBlur=0;

// === GATE MARKERS (ACTIVE BARREL) — Soulscape-style golden circles with wireframe ===
for(let k=0;k<actGT.length;k++){
const[gx,gy]=t(actGT[k][0],actGT[k][1]);const g=GI[k];
// Wireframe ring
ctx.beginPath();ctx.arc(gx,gy,14,0,Math.PI*2);ctx.strokeStyle='rgba(200,170,110,.3)';ctx.lineWidth=1;ctx.stroke();
ctx.beginPath();ctx.arc(gx,gy,10,0,Math.PI*2);ctx.strokeStyle='rgba(200,170,110,.15)';ctx.lineWidth=0.5;ctx.stroke();
// Golden core
ctx.save();ctx.shadowColor='#ffd700';ctx.shadowBlur=12;
ctx.beginPath();ctx.arc(gx,gy,6,0,Math.PI*2);ctx.fillStyle='#ffd700';ctx.fill();ctx.restore();
ctx.fillStyle='#1a1610';ctx.font='bold '+fs(26)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(k+1,gx,gy);
// Gate data label — Soulscape monospace style
const isFibT=SAL_FIB.has(k+1);
ctx.font=fs(24)+' monospace';ctx.fillStyle='rgba(200,169,110,.7)';ctx.textAlign='left';
ctx.fillText(g.spd.toFixed(1)+' AU/yr'+(isFibT?' F'+(k+1):''),gx+20,gy-6);
ctx.fillStyle='rgba(150,150,150,.5)';ctx.fillText('R:'+g.rsun.toFixed(2)+' T+'+g.tg.toFixed(2),gx+20,gy+18);
// Fibonacci ring
if(isFibT){ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=10;ctx.beginPath();ctx.arc(gx,gy,16,0,Math.PI*2);ctx.strokeStyle='rgba(255,170,60,.5)';ctx.lineWidth=2;ctx.stroke();ctx.restore()}
// Velocity direction arrow
const vAng=Math.atan2(-g.vy,g.vx);const vLen=14;
const vEx=gx+vLen*Math.cos(vAng),vEy=gy+vLen*Math.sin(vAng);
ctx.beginPath();ctx.moveTo(gx,gy);ctx.lineTo(vEx,vEy);ctx.strokeStyle='rgba(255,210,60,.6)';ctx.lineWidth=2;ctx.stroke();
const vAh=4;ctx.beginPath();ctx.moveTo(vEx,vEy);ctx.lineTo(vEx-vAh*Math.cos(vAng-.4),vEy-vAh*Math.sin(vAng-.4));
ctx.lineTo(vEx-vAh*Math.cos(vAng+.4),vEy-vAh*Math.sin(vAng+.4));ctx.closePath();ctx.fillStyle='rgba(255,210,60,.6)';ctx.fill()}
// Midpoint annotations between gates (curvature + speed change)
for(let mk=0;mk<actGT.length;mk++){const mg=GI[mk];
const prev2=mk===0?actB:actGT[mk-1];const mx3=(prev2[0]+actGT[mk][0])/2,my3=(prev2[1]+actGT[mk][1])/2;
const[mpx2,mpy2]=t(mx3,my3);
ctx.font=fs(14)+' sans-serif';ctx.textAlign='center';
ctx.fillStyle='rgba(220,200,140,.55)';ctx.fillText('\u2220'+Math.abs(mg.dhdg).toFixed(1)+'\u00b0',mpx2,mpy2-10);
ctx.fillStyle=mg.dspd>=0?'rgba(100,255,150,.5)':'rgba(255,130,100,.5)';
ctx.fillText((mg.dspd>=0?'+':'')+mg.dspd.toFixed(2)+' AU/yr',mpx2,mpy2+8)}

// === PARALLEL SCOPE overlay ===
if(showParallel&&PAR){
ctx.beginPath();ctx.strokeStyle='rgba(0,220,120,.4)';ctx.lineWidth=1.5;ctx.setLineDash([5,4]);
const pnt=PAR.nt;for(let pi=0;pi<pnt.length;pi++){const[psx,psy]=t(pnt[pi][0],pnt[pi][1]);if(pi===0)ctx.moveTo(psx,psy);else ctx.lineTo(psx,psy)}
ctx.stroke();ctx.setLineDash([]);
const[pbx2,pby2]=t(PAR.bp[0],PAR.bp[1]);
ctx.fillStyle='rgba(0,220,120,.5)';ctx.fillRect(pbx2-5,pby2-5,10,10);
ctx.font='bold '+fs(18)+' sans-serif';ctx.fillStyle='#00dd88';ctx.textAlign='center';ctx.fillText('PAR',pbx2,pby2+18);
// Diff arrows
for(let pk=0;pk<PAR.gt.length;pk++){
const[ppx,ppy]=t(GT[pk][0],GT[pk][1]);const[pqx,pqy]=t(PAR.gt[pk][0],PAR.gt[pk][1]);
ctx.beginPath();ctx.moveTo(ppx,ppy);ctx.lineTo(pqx,pqy);ctx.strokeStyle='rgba(0,255,140,.3)';ctx.lineWidth=1;ctx.stroke();
ctx.beginPath();ctx.arc(pqx,pqy,3,0,Math.PI*2);ctx.fillStyle='rgba(0,220,120,.5)';ctx.fill();
const pd=PAR.diff[pk];ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(0,255,160,.5)';ctx.textAlign='center';
ctx.fillText('\u0394'+pd.d_pos.toFixed(4),( ppx+pqx)/2,(ppy+pqy)/2-6)}
ctx.font='bold '+fs(16)+' monospace';ctx.fillStyle='#00dd88';ctx.textAlign='left';
ctx.fillText('PARALLEL (+'+PAR.off+'\u00b0) miss='+PAR.miss.toFixed(6)+' AU',14,h-90)}

// === SWARM BARRELS overlay ===
if(showSwarm&&SWM){
for(let si2=0;si2<SWM.length;si2++){const sw2=SWM[si2];const scol=SWM_COLS[si2%SWM_COLS.length];
ctx.beginPath();ctx.strokeStyle=scol+'66';ctx.lineWidth=1;ctx.setLineDash([3,5]);
const snt2=sw2.nt;for(let sj=0;sj<snt2.length;sj++){const[ssx,ssy]=t(snt2[sj][0],snt2[sj][1]);if(sj===0)ctx.moveTo(ssx,ssy);else ctx.lineTo(ssx,ssy)}
ctx.stroke();ctx.setLineDash([]);
const[sbx2,sby2]=t(sw2.bp[0],sw2.bp[1]);
ctx.beginPath();ctx.arc(sbx2,sby2,4,0,Math.PI*2);ctx.fillStyle=scol;ctx.fill();
ctx.font='bold '+fs(16)+' sans-serif';ctx.fillStyle=scol;ctx.textAlign='center';ctx.fillText('B@'+sw2.ck+'h',sbx2,sby2-12);
// Convergence line
ctx.beginPath();ctx.setLineDash([2,6]);ctx.strokeStyle=scol+'33';ctx.lineWidth=0.5;
const[stx2,sty2]=t(TG[0],TG[1]);ctx.moveTo(sbx2,sby2);ctx.lineTo(stx2,sty2);ctx.stroke();ctx.setLineDash([]);
// Swarm gate dots
for(let sk=0;sk<sw2.gt.length;sk++){const[sgx2,sgy2]=t(sw2.gt[sk][0],sw2.gt[sk][1]);
ctx.beginPath();ctx.arc(sgx2,sgy2,2,0,Math.PI*2);ctx.fillStyle=scol+'88';ctx.fill()}}
ctx.font='bold '+fs(16)+' monospace';ctx.fillStyle='#cc44ff';ctx.textAlign='left';
ctx.fillText('SWARM ('+SWM.length+' barrels) \u2192 TARGET',14,h-108)}

// === BARREL — Soulscape blue square ===
// === NON-PRIMARY BARRELS \u2014 dim trajectories + markers for context ===
// Click any of these markers (or the picker chips) to promote it to PRIMARY.
const _trjBarrelHits=[];
for(let bi2=0;bi2<allB_trj.length;bi2++){if(bi2===actIdxT)continue;const ob=allB_trj[bi2];
ctx.beginPath();ctx.strokeStyle=ob.col+'55';ctx.lineWidth=1;ctx.setLineDash([4,5]);
for(let i=0;i<ob.nt.length;i++){const[sx,sy]=t(ob.nt[i][0],ob.nt[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([]);
const[obx,oby]=t(ob.bp[0],ob.bp[1]);
ctx.beginPath();ctx.arc(obx,oby,6,0,Math.PI*2);ctx.fillStyle=ob.col+'88';ctx.fill();
ctx.strokeStyle='rgba(255,255,255,.35)';ctx.lineWidth=1;ctx.stroke();
ctx.font='bold '+fs(16)+' sans-serif';ctx.fillStyle=ob.col+'cc';ctx.textAlign='center';
ctx.fillText('NON-PRIMARY',obx,oby-14);
ctx.font=fs(14)+' sans-serif';ctx.fillStyle='#889';
ctx.fillText(ob.origLbl,obx,oby+18);
_trjBarrelHits.push({x:obx,y:oby,r:14,idx:bi2})}

// === ACTIVE BARREL \u2014 PRIMARY MARKER, tinted by the barrel color itself ===
const[bx,by]=t(actB[0],actB[1]);
ctx.save();ctx.shadowColor=active.col;ctx.shadowBlur=24;
ctx.fillStyle=active.col;ctx.fillRect(bx-9,by-9,18,18);ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(bx-9,by-9,18,18);
ctx.font='bold '+fs(30)+' "Times New Roman",serif';ctx.fillStyle=active.col;ctx.textAlign='center';
ctx.fillText('PRIMARY',bx,by+30);
ctx.font='bold '+fs(20)+' monospace';ctx.fillStyle='#c8a96e';
ctx.fillText(active.origLbl,bx,by+52);
ctx.font=fs(18)+' monospace';ctx.fillStyle='#889';
ctx.fillText('('+actB[0].toFixed(3)+', '+actB[1].toFixed(3)+') AU',bx,by+72);
_trjBarrelHits.push({x:bx,y:by,r:16,idx:actIdxT});

// === TARGET — ethereal green with pulsing ring ===
const[tgx,tgy]=t(TG[0],TG[1]);
const pulse=0.7+0.3*Math.sin(Date.now()*0.003);
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=30*pulse;
ctx.beginPath();ctx.arc(tgx,tgy,12,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,'+pulse+')';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tgx,tgy,12,0,Math.PI*2);ctx.stroke();
// Pulsing hit radius ring
ctx.beginPath();ctx.arc(tgx,tgy,HR*sc,0,Math.PI*2);ctx.strokeStyle='rgba(68,255,136,'+(0.2+0.15*pulse)+')';ctx.lineWidth=1;ctx.setLineDash([4,6]);ctx.stroke();ctx.setLineDash([]);
ctx.font='bold '+fs(32)+' "Times New Roman",serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';
ctx.fillText('TARGET',tgx,tgy-26);ctx.font=fs(24)+' monospace';ctx.fillText('('+TG[0].toFixed(2)+', '+TG[1].toFixed(2)+')',tgx,tgy+32);
// Gap line active-barrel → target
const gapDistT=Math.sqrt((TG[0]-actB[0])**2+(TG[1]-actB[1])**2);
ctx.beginPath();ctx.setLineDash([6,4]);ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1;
ctx.moveTo(bx,by);ctx.lineTo(tgx,tgy);ctx.stroke();ctx.setLineDash([]);
const gmxT=(bx+tgx)/2,gmyT=(by+tgy)/2;
ctx.font='bold '+fs(22)+' monospace';ctx.fillStyle='rgba(68,255,136,.5)';ctx.textAlign='center';
ctx.fillText(gapDistT.toFixed(3)+' AU',gmxT,gmyT-10);
// Target orbital info
if(TGT_ORB){ctx.font=fs(18)+' monospace';ctx.fillStyle='#88ccaa';ctx.textAlign='center';
ctx.fillText('v_orb='+TGT_ORB.spd+' AU/yr P='+TGT_ORB.period+' yr',tgx,tgy+54);
ctx.fillStyle='rgba(136,102,102,.6)';ctx.fillText('\u03b2='+TGT_ORB.beta.toExponential(2)+' \u03b3='+TGT_ORB.gamma.toFixed(8),tgx,tgy+72)}

// Legend — top-right (matching Scope)
ctx.save();ctx.globalAlpha=0.85;
ctx.fillStyle='rgba(10,10,26,.75)';ctx.fillRect(w-320,68,310,220);ctx.restore();
ctx.textAlign='left';ctx.font=fs(14)+' sans-serif';let llyT=84;
const legT=[['#ffcc00','\u25c7 FTOP (focal tensor origin)'],['rgba(120,170,255,.8)','Reference orbits (0.72,1.0,1.52 AU)'],['#ffd700','Correction gates with velocity arrows'],['#44ff88','Trajectory (speed-colored segments)'],['rgba(220,185,120,.7)','Scope ring / tensor map / clocks'],['#4488ff','Barrel (7 o\'clock / 240\u00b0)'],['rgba(255,170,60,.7)','Fibonacci gates (F1,F2,F3,F5,F8)'],['#00dd88','Parallel scope (P=toggle)'],['#cc44ff','Swarm barrels (S=toggle)'],['rgba(255,140,60,.5)','Target orbit history (orange)']];
for(const[col,lbl]of legT){ctx.fillStyle=col;ctx.fillRect(w-314,llyT,12,12);ctx.fillStyle='#aaa';ctx.fillText(lbl,w-298,llyT+10);llyT+=20}
// Hit/miss stats — bottom-right
ctx.textAlign='right';ctx.font='bold '+fs(20)+' monospace';
ctx.fillStyle='#44ff88';ctx.fillText('Hit:'+R.ch+'% | Miss:'+R.cmm+' AU',w-14,h-68);
ctx.fillStyle='#ff6644';ctx.fillText('Baseline:'+R.bh+'% | Miss:'+R.bmm+' AU',w-14,h-88);
ctx.fillStyle='#667';ctx.font=fs(16)+' sans-serif';ctx.fillText('Gap: '+gapDistT.toFixed(4)+' AU | |v\u2080|='+Math.sqrt(actV0[0]**2+actV0[1]**2).toFixed(3)+' AU/yr | Miss: '+active.miss.toExponential(2)+' AU | Click barrel to switch',w-14,h-108);

// === ANIMATED COMET — Soulscape ember particles (ACTIVE BARREL path) ===
if(animT>0){
const f=animT/TF;const idx=f*(actNT.length-1);const i0=Math.floor(idx),i1=Math.min(i0+1,actNT.length-1);const tt3=idx-i0;
const ax=actNT[i0][0]+(actNT[i1][0]-actNT[i0][0])*tt3,ay=actNT[i0][1]+(actNT[i1][1]-actNT[i0][1])*tt3;
const[asx,asy]=t(ax,ay);
// Spawn ember particles
for(let pp=0;pp<3;pp++){TRJ_PARTICLES.push({x:asx+Math.random()*6-3,y:asy+Math.random()*6-3,vx:Math.random()*2-1,vy:Math.random()*2-1,life:1,maxLife:30+Math.random()*20,
col:Math.random()>0.5?'255,150,40':'255,200,80'})}
// Draw & update particles
for(let i=TRJ_PARTICLES.length-1;i>=0;i--){const p=TRJ_PARTICLES[i];
p.x+=p.vx;p.y+=p.vy;p.vy-=0.02;p.life++;
const al=Math.max(0,1-p.life/p.maxLife);
ctx.beginPath();ctx.arc(p.x,p.y,2*al,0,Math.PI*2);ctx.fillStyle='rgba('+p.col+','+al+')';ctx.fill();
if(p.life>p.maxLife)TRJ_PARTICLES.splice(i,1)}
// Comet trail
for(let tt2=1;tt2<=20;tt2++){const ft=Math.max(0,f-tt2*0.002);const ii=ft*(actNT.length-1);const ti0=Math.floor(ii),ti1=Math.min(ti0+1,actNT.length-1);const tfr=ii-ti0;
const tx2=actNT[ti0][0]+(actNT[ti1][0]-actNT[ti0][0])*tfr,ty2=actNT[ti0][1]+(actNT[ti1][1]-actNT[ti0][1])*tfr;
const[px2,py2]=t(tx2,ty2);const al=0.7-tt2*0.035;
ctx.beginPath();ctx.arc(px2,py2,5-tt2*0.2,0,Math.PI*2);ctx.fillStyle='rgba(255,150,40,'+Math.max(0,al)+')';ctx.fill()}
// Comet head — bright core with bloom
ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=35;
ctx.beginPath();ctx.arc(asx,asy,8,0,Math.PI*2);ctx.fillStyle='#ffaa00';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(asx,asy,8,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
// Inner bright core
ctx.beginPath();ctx.arc(asx,asy,3,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
// Distance to target line
const dTgt=Math.sqrt((ax-TG[0])**2+(ay-TG[1])**2);
ctx.beginPath();ctx.setLineDash([4,6]);ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1;
ctx.moveTo(asx,asy);ctx.lineTo(tgx,tgy);ctx.stroke();ctx.setLineDash([]);
ctx.font=fs(24)+' monospace';ctx.fillStyle='rgba(68,255,136,.5)';ctx.textAlign='center';
ctx.fillText(dTgt.toFixed(4)+' AU',( asx+tgx)/2,(asy+tgy)/2-10)}

// === WOW-STYLE HUD FRAME (top-left) ===
const gk=Math.min(Math.floor(animT/DTG),11);const gData=GI[gk];
const spd=animT>0?getAnimSpd():0;const gate=animT>0?getAnimGate():0;
let hd='<div style="font:bold 12px \'Times New Roman\',serif;color:#c8a96e;letter-spacing:2px;margin-bottom:6px;border-bottom:1px solid #5a4a32;padding-bottom:4px">TRAJECTORY — PRIMARY</div>';
hd+='<div style="display:flex;gap:8px;margin-bottom:6px;align-items:center">';
hd+='<div style="font-size:22px;line-height:1">&#9790;</div>';
hd+='<div><div style="color:'+active.col+';font:bold 11px \'Times New Roman\',serif">'+active.origLbl+'</div>';
hd+='<div style="color:#889;font-size:9px">T+'+animT.toFixed(3)+' yr of '+TF+' yr</div></div></div>';
hd+='<div style="font-size:9px;color:#889;margin-bottom:4px">Barrel ('+actB[0].toFixed(3)+', '+actB[1].toFixed(3)+') | |v₀|='+Math.sqrt(actV0[0]**2+actV0[1]**2).toFixed(3)+' | Miss '+active.miss.toExponential(2)+' AU</div>';
// Progress bar — Soulscape HP bar style
const pct=TF>0?(animT/TF*100):0;
hd+='<div style="height:12px;background:rgba(0,0,0,.8);border:1px solid #444;border-radius:3px;overflow:hidden;margin-bottom:4px">';
hd+='<div style="width:'+pct+'%;height:100%;background:linear-gradient(180deg,#44cc44,#228822);border-radius:2px"></div></div>';
hd+='<div style="font-size:9px;color:#889;margin-bottom:6px">'+pct.toFixed(1)+'% complete</div>';
if(animT>0){
hd+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;font-size:9px">';
hd+='<span style="color:#aa9">Speed</span><span style="color:#ffd700">'+spd.toFixed(2)+' AU/yr</span>';
hd+='<span style="color:#aa9">Gate</span><span style="color:#ffd700">'+gate+'/12</span>';
hd+='<span style="color:#aa9">R(FTOP)</span><span style="color:#ffd700">'+gData.rsun+' AU</span>';
hd+='<span style="color:#aa9">Heading</span><span style="color:#ffd700">'+gData.hdg+'\u00b0</span>';
hd+='<span style="color:#aa9">Curvature</span><span style="color:#ffd700">'+gData.curv+' /AU</span>';
hd+='<span style="color:#aa9">Gravity</span><span style="color:#ffd700">'+gData.grav+' AU/yr\u00b2</span>';
hd+='</div>'}
document.getElementById('trjHud').innerHTML=hd;
// Picker chips — clicking sets activeBarrel via salvoFocus and redraws.
{const pk=document.getElementById('trjPicker');if(pk){let ph='';
for(let bi2=0;bi2<allB_trj.length;bi2++){const b=allB_trj[bi2];const isAct=bi2===actIdxT;
ph+='<button onclick="salvoFocus='+bi2+';draw()" title="Click to make '+b.origLbl+' the PRIMARY view" '+
'style="font:bold 10px monospace;padding:4px 8px;border:2px solid '+b.col+';border-radius:4px;cursor:pointer;'+
'background:'+(isAct?b.col:'rgba(10,10,26,.85)')+';'+
'color:'+(isAct?'#0a0a0a':b.col)+';min-width:60px">'+
(isAct?'◆ ':'')+b.origLbl+'</button>'}
pk.innerHTML=ph}}
// Bottom info bar \u2014 shows who's the active PRIMARY barrel in this view
let bi='<span style="color:'+active.col+'"><b>PRIMARY:</b> '+active.origLbl+'</span> \u2022 ';
bi+='<span style="color:#ffd700">'+R.ch+'%</span> hit rate \u2022 ';
bi+='<span style="color:#889">Gap: '+gapDistT.toFixed(3)+' AU</span> \u2022 ';
bi+='<span style="color:#889">Miss: '+active.miss.toExponential(2)+' AU</span> \u2022 ';
bi+='<span style="color:#556">SPACE=play \u2022 \u2190\u2192=step \u2022 Click barrel/chip to switch</span>';
document.getElementById('trjInfo').innerHTML=bi;
// Store screen positions for hover detection (active barrel's gates + trajectory)
const trjC2=document.getElementById('trjC');
const _trjGates=[];
for(let k=0;k<actGT.length;k++){const[gx2,gy2]=t(actGT[k][0],actGT[k][1]);_trjGates.push({sx:gx2,sy:gy2,k:k,g:GI[k]})}
const _trjTraj=[];const _trjStep=Math.max(1,Math.floor(actNT.length/400));
for(let i=0;i<actNT.length;i+=_trjStep){const[tx2,ty2]=t(actNT[i][0],actNT[i][1]);_trjTraj.push({sx:tx2,sy:ty2,wx:actNT[i][0],wy:actNT[i][1],ti:i/(actNT.length-1)*TF})}
trjC2._gates=_trjGates;trjC2._traj=_trjTraj;trjC2._barrels=_trjBarrelHits;
// Wire hover once
if(!trjC2._hoverWired){trjC2._hoverWired=true;
trjC2.style.cursor='pointer';
// Click any barrel marker to make it the active PRIMARY of this tab (and, via
// the shared salvoFocus state, of every other tab too).
trjC2.addEventListener('click',e=>{
const rc=trjC2.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const barrels=trjC2._barrels||[];let pick=-1,bestB=9999;
for(const bh of barrels){const dx=mx-bh.x,dy=my-bh.y,d=dx*dx+dy*dy;if(d<bh.r*bh.r&&d<bestB){bestB=d;pick=bh.idx}}
if(pick>=0){salvoFocus=pick;draw()}});
trjC2.addEventListener('mousemove',e=>{
const rc=trjC2.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const ttip=document.getElementById('trjTip');
// Priority: barrel markers (so clicking is discoverable), then gates, then trajectory
const barrels=trjC2._barrels||[];let foundT=null,bestDT=1600;
for(const bh of barrels){const dx=mx-bh.x,dy=my-bh.y,d=dx*dx+dy*dy;if(d<bh.r*bh.r&&d<bestDT){bestDT=d;foundT={type:'barrel',idx:bh.idx}}}
if(!foundT){const gates=trjC2._gates||[];for(const g of gates){const dx=mx-g.sx,dy=my-g.sy;const d=dx*dx+dy*dy;if(d<bestDT){bestDT=d;foundT={type:'gate',g:g.g,k:g.k}}}}
if(!foundT||bestDT>400){const trj=trjC2._traj||[];for(const p of trj){const dx=mx-p.sx,dy=my-p.sy;const d=dx*dx+dy*dy;if(d<bestDT){bestDT=d;foundT={type:'traj',p:p}}}}
if(foundT&&bestDT<900){
if(foundT.type==='barrel'){const bref=(typeof buildAllBarrels==='function'?buildAllBarrels():[])[foundT.idx];
if(bref){const vm=Math.sqrt(bref.v0[0]*bref.v0[0]+bref.v0[1]*bref.v0[1]);
const roleLbl=bref.isActive?'PRIMARY (active)':'NON-PRIMARY';
const roleCol=bref.isActive?bref.col:'#889';
ttip.innerHTML='<b style="color:'+roleCol+'">'+roleLbl+'</b><br>'+
'<b style="color:'+bref.col+'">'+bref.origLbl+'</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+bref.bp[0].toFixed(4)+', '+bref.bp[1].toFixed(4)+') AU</span><br>'+
'<span style="color:#889">|v₀|:</span> <span style="color:#ffd700">'+vm.toFixed(4)+' AU/yr</span><br>'+
'<span style="color:#889">Miss:</span> <span style="color:'+(bref.miss<0.1?'#44ff88':'#ff4444')+'">'+bref.miss.toExponential(2)+' AU</span><br>'+
(bref.proj?'<span style="color:#889">⌀:</span> <span style="color:#ffd700">'+bref.proj.diameter_m+' m</span> | <span style="color:#889">m:</span> <span style="color:#ffd700">'+bref.proj.mass_kg.toExponential(2)+' kg</span><br>'+
'<span style="color:#889">KE:</span> <span style="color:#ffaa44">'+bref.proj.ke_j.toExponential(2)+' J</span> ('+bref.proj.tnt_kt.toFixed(2)+' kt)<br>':'')+
'<span style="color:#556">Click to make this barrel the PRIMARY view</span>'}
else{ttip.innerHTML='<b>Barrel</b>'}}
else if(foundT.type==='gate'){const g=foundT.g;const isFib=SAL_FIB.has(foundT.k+1);
const sa=sphereAddr(g.pos[0],g.pos[1]);
ttip.innerHTML='<b style="color:#ffd700">'+(isFib?'\u2605 Fib ':'')+'Gate '+(foundT.k+1)+'</b><br>'+
'<span style="color:#889">Pos:</span> <span style="color:#ffd700">('+g.pos[0]+','+g.pos[1]+') AU</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">Speed:</span> <span style="color:#ffd700">'+g.spd+' AU/yr</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+g.rsun+' AU</span><br>'+
'<span style="color:#889">Heading:</span> <span style="color:#ffd700">'+g.hdg+'\u00b0</span> \u00b7 <span style="color:#889">\u0394hdg:</span> <span style="color:'+(g.dhdg>=0?'#44ff88':'#ff8866')+'">'+(g.dhdg>=0?'+':'')+g.dhdg+'\u00b0</span><br>'+
'<span style="color:#889">\u0394Spd:</span> <span style="color:'+(g.dspd>=0?'#44ff88':'#ff8866')+'">'+(g.dspd>=0?'+':'')+g.dspd+' AU/yr</span><br>'+
'<span style="color:#889">Curvature:</span> <span style="color:#ffd700">'+g.curv+' /AU</span><br>'+
'<span style="color:#889">Arc:</span> <span style="color:#ffd700">'+g.arc+' AU</span><br>'+
'<span style="color:#889">J-cond:</span> <span style="color:#ffd700">'+g.jc+'</span><br>'+
'<span style="color:#889">T+</span><span style="color:#ffd700">'+g.tg+' yr</span> \u00b7 <span style="color:#889">t<sub>go</sub>:</span> <span style="color:#ffd700">'+g.tr+'</span>'}
else{const p=foundT.p;const rr=Math.sqrt(p.wx*p.wx+p.wy*p.wy);
const dTgt=Math.sqrt((p.wx-TG[0])**2+(p.wy-TG[1])**2);
const sa=sphereAddr(p.wx,p.wy);
ttip.innerHTML='<b style="color:#44ff88">Trajectory sample</b><br>'+
'<span style="color:#889">Pos:</span> <span style="color:#ffd700">('+p.wx.toFixed(4)+','+p.wy.toFixed(4)+') AU</span><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+sa.addr+'</span> <span style="color:#889">('+(sa.inScope?'in':'out')+')</span><br>'+
'<span style="color:#889">Cycle:</span> <span style="color:#ffd700">'+sa.time+'</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+rr.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">D(Target):</span> <span style="color:'+(dTgt<0.5?'#44ff88':'#ffd700')+'">'+dTgt.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+p.ti.toFixed(3)+' yr</span>'}
ttip.style.display='block';ttip.style.left=Math.min(mx+16,rc.width-320)+'px';ttip.style.top=Math.max(my-120,10)+'px'}
else{ttip.style.display='none'}});
trjC2.addEventListener('mouseleave',()=>{document.getElementById('trjTip').style.display='none'})}
// Request next frame if playing (particle animation)
if(playing)requestAnimationFrame(drawTrajectory)}

// ═══ 3D SPHERE NAVIGATION TAB — transparent sphere with internal 3D grid ═══
// The sphere is a transparent glass ball. FTOP is at center (0,0,0).
// Objects are at their actual grid coords INSIDE the sphere.
// The sphere shell at r=30 is a thin transparent boundary.
// Grid planes (XY, XZ, YZ) are visible through the glass.
function draw3DSphere(){
const c=document.getElementById('s3dC');const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
const cx=w/2,cy=h/2;
const sc=Math.min(w,h)/72*s3z; // scale: grid unit → pixels (±30 fills sphere)
ctx.fillStyle='#040408';ctx.fillRect(0,0,w,h);
// Stars
for(let i=0;i<80;i++){const sx2=((i*137.5+42)%w),sy2=((i*97.3+17)%h);const br=0.12+Math.sin(Date.now()*0.001+i)*0.08;
ctx.beginPath();ctx.arc(sx2,sy2,0.6,0,Math.PI*2);ctx.fillStyle='rgba(200,180,150,'+br+')';ctx.fill()}

// === 3D ROTATION + PROJECTION ===
const cosRx=Math.cos(s3rx),sinRx=Math.sin(s3rx),cosRy=Math.cos(s3ry),sinRy=Math.sin(s3ry);
function p3d(x,y,z){
let x2=x*cosRy+z*sinRy,z2=-x*sinRy+z*cosRy;
let y2=y*cosRx-z2*sinRx,z3=y*sinRx+z2*cosRx;
return[cx+x2*sc,cy-y2*sc,z3]}
// AU → grid (inside sphere, z=0 for 2D data)
function au2g(ax,ay){return[ax/Rs*30,ay/Rs*30]}

// === TRANSPARENT SPHERE SHELL (r=30, thin outline) ===
const sR=30*sc; // sphere pixel radius
// Back hemisphere hint (very faint)
ctx.beginPath();ctx.arc(cx,cy,sR,0,Math.PI*2);
ctx.fillStyle='rgba(15,15,30,0.15)';ctx.fill();
// Sphere outline
ctx.beginPath();ctx.arc(cx,cy,sR,0,Math.PI*2);
ctx.strokeStyle='rgba(200,169,110,0.3)';ctx.lineWidth=1.5;ctx.stroke();

// === 3D INTERNAL GRID (three axis-aligned planes visible through glass) ===
// XY plane grid (z=0) — the main data plane
ctx.strokeStyle='rgba(200,169,110,.12)';ctx.lineWidth=0.4;
for(let g=-30;g<=30;g+=10){
// Lines along X (constant Y=g, z=0) — clip to sphere
const pts=[];for(let t=-30;t<=30;t+=2){if(Math.sqrt(t*t+g*g)<=30.5)pts.push(p3d(t,g,0))}
if(pts.length>1){ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let j=1;j<pts.length;j++)ctx.lineTo(pts[j][0],pts[j][1]);ctx.stroke()}
// Lines along Y (constant X=g, z=0)
const pts2=[];for(let t=-30;t<=30;t+=2){if(Math.sqrt(g*g+t*t)<=30.5)pts2.push(p3d(g,t,0))}
if(pts2.length>1){ctx.beginPath();ctx.moveTo(pts2[0][0],pts2[0][1]);for(let j=1;j<pts2.length;j++)ctx.lineTo(pts2[j][0],pts2[j][1]);ctx.stroke()}}
// XZ plane grid (y=0) — vertical cross-section
ctx.strokeStyle='rgba(140,190,255,.28)';ctx.lineWidth=0.6;
for(let g=-30;g<=30;g+=10){
const pts=[];for(let t=-30;t<=30;t+=2){if(Math.sqrt(t*t+g*g)<=30.5)pts.push(p3d(t,0,g))}
if(pts.length>1){ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let j=1;j<pts.length;j++)ctx.lineTo(pts[j][0],pts[j][1]);ctx.stroke()}
const pts2=[];for(let t=-30;t<=30;t+=2){if(Math.sqrt(g*g+t*t)<=30.5)pts2.push(p3d(g,0,t))}
if(pts2.length>1){ctx.beginPath();ctx.moveTo(pts2[0][0],pts2[0][1]);for(let j=1;j<pts2.length;j++)ctx.lineTo(pts2[j][0],pts2[j][1]);ctx.stroke()}}
// YZ plane grid (x=0) — vertical cross-section
ctx.strokeStyle='rgba(130,230,170,.22)';ctx.lineWidth=0.6;
for(let g=-30;g<=30;g+=10){
const pts=[];for(let t=-30;t<=30;t+=2){if(Math.sqrt(t*t+g*g)<=30.5)pts.push(p3d(0,t,g))}
if(pts.length>1){ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);for(let j=1;j<pts.length;j++)ctx.lineTo(pts[j][0],pts[j][1]);ctx.stroke()}
const pts2=[];for(let t=-30;t<=30;t+=2){if(Math.sqrt(g*g+t*t)<=30.5)pts2.push(p3d(0,g,t))}
if(pts2.length>1){ctx.beginPath();ctx.moveTo(pts2[0][0],pts2[0][1]);for(let j=1;j<pts2.length;j++)ctx.lineTo(pts2[j][0],pts2[j][1]);ctx.stroke()}}

// === MAJOR AXES through center (bold) ===
// X axis (amber)
{const[x1,y1]=p3d(-30,0,0);const[x2,y2]=p3d(30,0,0);
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle='rgba(200,169,110,.3)';ctx.lineWidth=1.5;ctx.stroke();
const[lx,ly]=p3d(32,0,0);ctx.font='bold '+fs(18)+' monospace';ctx.fillStyle='rgba(200,169,110,.5)';ctx.textAlign='center';ctx.fillText('X',lx,ly)}
// Y axis (amber)
{const[x1,y1]=p3d(0,-30,0);const[x2,y2]=p3d(0,30,0);
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle='rgba(200,169,110,.3)';ctx.lineWidth=1.5;ctx.stroke();
const[lx,ly]=p3d(0,32,0);ctx.font='bold '+fs(18)+' monospace';ctx.fillStyle='rgba(200,169,110,.5)';ctx.textAlign='center';ctx.fillText('Y',lx,ly)}
// Z axis (blue, depth)
{const[x1,y1]=p3d(0,0,-30);const[x2,y2]=p3d(0,0,30);
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle='rgba(120,170,255,.25)';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([]);
const[lx,ly]=p3d(0,0,33);ctx.font='bold '+fs(18)+' monospace';ctx.fillStyle='rgba(120,170,255,.4)';ctx.textAlign='center';ctx.fillText('Z',lx,ly)}

// Axis tick labels along X
ctx.font=fs(12)+' monospace';ctx.fillStyle='rgba(200,170,110,.3)';ctx.textAlign='center';
for(let g=-30;g<=30;g+=10){if(g===0)continue;const[lx,ly]=p3d(g,-2,0);ctx.fillText(g,lx,ly)}

// === EQUATORIAL RING (XY plane circle at r=30) — sphere boundary ===
ctx.beginPath();ctx.strokeStyle='rgba(200,169,110,0.35)';ctx.lineWidth=1.5;
for(let a=0;a<=360;a+=3){const[sx2,sy2]=p3d(30*Math.cos(a*Math.PI/180),30*Math.sin(a*Math.PI/180),0);
if(a===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();
// Vertical great circles at 0°, 90° — sphere wireframe
for(let ang of[0,90]){const ar=ang*Math.PI/180;
ctx.beginPath();ctx.strokeStyle='rgba(200,169,110,0.12)';ctx.lineWidth=0.7;
for(let b=0;b<=360;b+=3){const br2=b*Math.PI/180;
const[sx2,sy2]=p3d(30*Math.cos(br2)*Math.cos(ar),30*Math.cos(br2)*Math.sin(ar),30*Math.sin(br2));
if(b===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke()}
// Additional meridian circles every 30°
for(let ang=30;ang<360;ang+=30){if(ang===90||ang===180||ang===270)continue;
const ar=ang*Math.PI/180;
ctx.beginPath();ctx.strokeStyle='rgba(200,169,110,0.06)';ctx.lineWidth=0.4;
for(let b=0;b<=360;b+=5){const br2=b*Math.PI/180;
const[sx2,sy2]=p3d(30*Math.cos(br2)*Math.cos(ar),30*Math.cos(br2)*Math.sin(ar),30*Math.sin(br2));
if(b===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke()}

// === DEGREE TICKS (0°-360° on equatorial ring) ===
for(let d=0;d<360;d+=30){const a=d*Math.PI/180;
const[tx,ty]=p3d(32*Math.cos(a),32*Math.sin(a),0);
ctx.font='bold '+fs(14)+' monospace';ctx.fillStyle='rgba(200,169,110,.35)';ctx.textAlign='center';ctx.fillText(d+'\u00b0',tx,ty)}

// === CLOCK NUMBERS on equatorial ring ===
for(let ci=1;ci<=12;ci++){const ca=(90-ci*30)*Math.PI/180;
const[sx2,sy2]=p3d(34*Math.cos(ca),34*Math.sin(ca),0);
ctx.font='bold '+fs(16)+' sans-serif';ctx.fillStyle=ci===7?'rgba(100,170,255,.6)':'rgba(160,144,112,.3)';
ctx.textAlign='center';ctx.fillText(ci,sx2,sy2)}

// Transparent 3D Flower-of-Life sphere grid — the full 3D expansion of the
// 2D stencil, drawn inside the dimensional sphere. AU coords scale to grid
// units via /Rs*30 so the lattice sits in the same space as the gates + target.
if(DV.X){const to3d=(x,y,z)=>{const p=p3d(x/Rs*30,y/Rs*30,z/Rs*30);return[p[0],p[1]]};
drawFoL3DSpheres(ctx,to3d,{alpha:0.22,use3D:true})}

// === REFERENCE ORBIT RINGS (on XY plane, inside sphere) ===
const orbits=[[0.72,'rgba(210,180,100,.2)','Venus'],[1.0,'rgba(120,170,255,.25)','Earth'],[1.52,'rgba(255,120,90,.2)','Mars']];
for(const[rau,col,lbl]of orbits){const gr=rau/Rs*30;
ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=1;ctx.setLineDash([4,4]);
for(let a=0;a<=360;a+=5){const[sx2,sy2]=p3d(gr*Math.cos(a*Math.PI/180),gr*Math.sin(a*Math.PI/180),0);
if(a===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();ctx.setLineDash([]);
const[olx,oly]=p3d(gr+2,1,0);ctx.font=fs(12)+' sans-serif';ctx.fillStyle=col;ctx.textAlign='left';ctx.fillText(lbl+' '+rau,olx,oly)}

// === FTOP at center (0,0,0) ===
const[ox,oy]=p3d(0,0,0);
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=18;
ctx.beginPath();ctx.arc(ox,oy,7,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
ctx.font='bold '+fs(20)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('\u25c7 FTOP (0,0)',ox,oy-16);
ctx.font=fs(13)+' monospace';ctx.fillStyle='#aa8';ctx.fillText('0\u00b0,0z,0x,0y',ox,oy+22);

// === TRAJECTORY (speed-colored, inside sphere at z=0) ===
const allSpd=GI.map(g=>g.spd);const mnS=Math.min(...allSpd),mxS=Math.max(...allSpd),rngS=mxS-mnS+.001;
const nPerSeg=Math.floor(NT.length/13);
// Glow
ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.06)';ctx.lineWidth=7;
for(let i=0;i<NT.length;i+=3){const[gx,gy]=au2g(NT[i][0],NT[i][1]);const[sx2,sy2]=p3d(gx,gy,0);
if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();
// Speed-colored segments
for(let seg=0;seg<13;seg++){
const colIdx=Math.min(seg,11);const spd=seg<12?GI[colIdx].spd:GI[11].spd;
const tt=(spd-mnS)/rngS;const col='rgb('+Math.round(80+tt*175)+','+Math.round(255-tt*80)+','+Math.round(200-tt*180)+')';
ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.shadowColor=col;ctx.shadowBlur=5;
const i0=seg*nPerSeg,i1=Math.min((seg+1)*nPerSeg+1,NT.length);
for(let i=i0;i<i1;i+=2){const[gx,gy]=au2g(NT[i][0],NT[i][1]);const[sx2,sy2]=p3d(gx,gy,0);
if(i===i0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke()}ctx.shadowBlur=0;

// === GATES (inside sphere at their grid positions) ===
const _s3dGates=[];
for(let k=0;k<GT.length;k++){const[gx,gy]=au2g(GT[k][0],GT[k][1]);const[sx2,sy2,dz]=p3d(gx,gy,0);
_s3dGates.push({sx:sx2,sy:sy2,k:k,g:GI[k],vis:true});
const isFib=SAL_FIB.has(k+1);
if(isFib){ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=10;ctx.beginPath();ctx.arc(sx2,sy2,12,0,Math.PI*2);
ctx.strokeStyle='rgba(255,170,60,.5)';ctx.lineWidth=2;ctx.stroke();ctx.restore()}
ctx.beginPath();ctx.arc(sx2,sy2,8,0,Math.PI*2);ctx.fillStyle='#ffd700';ctx.fill();
ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();
ctx.fillStyle='#1a1610';ctx.font='bold '+fs(18)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(k+1,sx2,sy2);
const gsph=au2sph(GT[k][0],GT[k][1]);
ctx.font=fs(13)+' monospace';ctx.fillStyle='rgba(255,215,0,.5)';ctx.textBaseline='alphabetic';ctx.textAlign='left';
ctx.fillText(gsph.addr+' '+GI[k].spd.toFixed(1)+(isFib?' F'+(k+1):''),sx2+14,sy2-4);
// Velocity arrow
const g=GI[k];const vAng=Math.atan2(-g.vy,g.vx);const vLen=12;
const vEx=sx2+vLen*Math.cos(vAng),vEy=sy2+vLen*Math.sin(vAng);
ctx.beginPath();ctx.moveTo(sx2,sy2);ctx.lineTo(vEx,vEy);ctx.strokeStyle='rgba(255,210,60,.5)';ctx.lineWidth=1.5;ctx.stroke()}

// === BARREL (inside sphere) ===
const[bgx,bgy]=au2g(B[0],B[1]);const[bsx,bsy]=p3d(bgx,bgy,0);const bsph=au2sph(B[0],B[1]);
ctx.save();ctx.shadowColor='#4488ff';ctx.shadowBlur=15;
ctx.fillStyle='#4488ff';ctx.fillRect(bsx-7,bsy-7,14,14);ctx.restore();
ctx.strokeStyle='#88bbff';ctx.lineWidth=2;ctx.strokeRect(bsx-7,bsy-7,14,14);
ctx.font='bold '+fs(18)+' sans-serif';ctx.fillStyle='#88bbff';ctx.textAlign='center';
ctx.fillText('BARREL 7h',bsx,bsy+24);
ctx.font=fs(13)+' monospace';ctx.fillStyle='#66aaff';ctx.fillText(bsph.addr,bsx,bsy+40);

// === TARGET (inside sphere, optionally off-plane via TGT_Z) ===
// Convert the target's world-AU (x, y, z) to sphere-grid (±30 per axis).
const[tgx,tgy]=au2g(TG[0],TG[1]);
const tgz_g=(TGT_Z||0)/Math.max(Rs,1e-9)*30;
const[tsx,tsy]=p3d(tgx,tgy,tgz_g);
const tsph=au2sph(TG[0],TG[1],TGT_Z||0);
const pulse=0.7+0.3*Math.sin(Date.now()*0.003);
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=20*pulse;
ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,'+pulse+')';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.stroke();
// If target has a z-component, draw a vertical drop line to the XY plane
if(Math.abs(tgz_g)>1e-3){const[txg0,tyg0]=p3d(tgx,tgy,0);
ctx.beginPath();ctx.strokeStyle='rgba(136,255,204,.6)';ctx.lineWidth=1.5;ctx.setLineDash([2,3]);
ctx.moveTo(txg0,tyg0);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([])}
ctx.font='bold '+fs(18)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';
ctx.fillText('TARGET',tsx,tsy-22);
ctx.font=fs(13)+' monospace';ctx.fillStyle='#44ff88';ctx.fillText(tsph.addr,tsx,tsy+28);
// Hit radius circle
const hrg=HR/Rs*30;
ctx.beginPath();for(let a=0;a<=360;a+=5){const[hx,hy]=p3d(tgx+hrg*Math.cos(a*Math.PI/180),tgy+hrg*Math.sin(a*Math.PI/180),0);
if(a===0)ctx.moveTo(hx,hy);else ctx.lineTo(hx,hy)}
ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1;ctx.setLineDash([3,5]);ctx.stroke();ctx.setLineDash([]);
// Gap line barrel→target
ctx.beginPath();ctx.setLineDash([4,6]);ctx.strokeStyle='rgba(68,255,136,.12)';ctx.lineWidth=1;
ctx.moveTo(bsx,bsy);ctx.lineTo(tsx,tsy);ctx.stroke();ctx.setLineDash([]);

// === TARGET ORBITAL PATH (inside sphere) ===
if(TGT_ORB&&TGT_ORB.history){ctx.beginPath();ctx.strokeStyle='rgba(255,140,60,.18)';ctx.lineWidth=1.5;ctx.setLineDash([3,5]);
const hSt=Math.max(1,Math.floor(TGT_ORB.history.length/200));
for(let i=0;i<TGT_ORB.history.length;i+=hSt){const[gx,gy]=au2g(TGT_ORB.history[i][0],TGT_ORB.history[i][1]);
const[sx2,sy2]=p3d(gx,gy,0);if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();ctx.setLineDash([])}
if(TGT_ORB&&TGT_ORB.path){ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1.5;ctx.setLineDash([5,5]);
const fSt=Math.max(1,Math.floor(TGT_ORB.path.length/200));
for(let i=0;i<TGT_ORB.path.length;i+=fSt){const[gx,gy]=au2g(TGT_ORB.path[i][0],TGT_ORB.path[i][1]);
const[sx2,sy2]=p3d(gx,gy,0);if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();ctx.setLineDash([]);
const[tfgx,tfgy]=au2g(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);const[tfsx,tfsy]=p3d(tfgx,tfgy,0);
ctx.beginPath();ctx.arc(tfsx,tfsy,5,0,Math.PI*2);ctx.strokeStyle='#44ff88';ctx.lineWidth=1.5;ctx.setLineDash([2,2]);ctx.stroke();ctx.setLineDash([]);
const tfSph=au2sph(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
ctx.font=fs(12)+' monospace';ctx.fillStyle='rgba(68,255,136,.5)';ctx.textAlign='center';ctx.fillText('T@Tf '+tfSph.addr,tfsx,tfsy+14)}

// === PARALLEL SCOPE (inside sphere) ===
if(showParallel&&PAR){ctx.beginPath();ctx.strokeStyle='rgba(0,220,120,.4)';ctx.lineWidth=1.5;ctx.setLineDash([5,4]);
for(let i=0;i<PAR.nt.length;i++){const[gx,gy]=au2g(PAR.nt[i][0],PAR.nt[i][1]);const[sx2,sy2]=p3d(gx,gy,0);
if(i===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();ctx.setLineDash([]);
const[pgx,pgy]=au2g(PAR.bp[0],PAR.bp[1]);const[psx,psy]=p3d(pgx,pgy,0);
ctx.fillStyle='rgba(0,220,120,.5)';ctx.fillRect(psx-5,psy-5,10,10);
ctx.font='bold '+fs(14)+' sans-serif';ctx.fillStyle='#00dd88';ctx.textAlign='center';ctx.fillText('PAR',psx,psy-12)}

// === SWARM BARRELS (inside sphere) ===
if(showSwarm&&SWM){for(let si=0;si<SWM.length;si++){const sw=SWM[si];const scol=SWM_COLS[si%SWM_COLS.length];
ctx.beginPath();ctx.strokeStyle=scol+'88';ctx.lineWidth=1;ctx.setLineDash([3,5]);
for(let j=0;j<sw.nt.length;j++){const[gx,gy]=au2g(sw.nt[j][0],sw.nt[j][1]);const[sx2,sy2]=p3d(gx,gy,0);
if(j===0)ctx.moveTo(sx2,sy2);else ctx.lineTo(sx2,sy2)}ctx.stroke();ctx.setLineDash([]);
const[sgx,sgy]=au2g(sw.bp[0],sw.bp[1]);const[ssx,ssy]=p3d(sgx,sgy,0);
ctx.beginPath();ctx.arc(ssx,ssy,4,0,Math.PI*2);ctx.fillStyle=scol;ctx.fill();
ctx.font='bold '+fs(14)+' sans-serif';ctx.fillStyle=scol;ctx.textAlign='center';ctx.fillText('S@'+sw.ck+'h',ssx,ssy-10)}}

// === ANIMATED COMET (inside sphere) ===
if(animT>0){const ap=getAnimPos();const[agx,agy]=au2g(ap[0],ap[1]);const[asx,asy]=p3d(agx,agy,0);
ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=25;
ctx.beginPath();ctx.arc(asx,asy,8,0,Math.PI*2);ctx.fillStyle='#ffaa00';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(asx,asy,8,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
ctx.beginPath();ctx.arc(asx,asy,3,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
const csph=au2sph(ap[0],ap[1]);
ctx.font='bold '+fs(18)+' monospace';ctx.fillStyle='#ffaa00';ctx.textAlign='left';
ctx.fillText('T+'+animT.toFixed(3)+' yr',asx+14,asy);
ctx.font=fs(13)+' monospace';ctx.fillStyle='#ffd700';ctx.fillText(csph.addr+' | '+csph.time,asx+14,asy+18)}

// === TITLE ===
ctx.font='bold '+fs(28)+' "Times New Roman",serif';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
ctx.fillText('3D Dimensional Sphere \u2014 X\u00b0,xz,xx,xy \u2014 Internal Grid',w/2,30);
ctx.font=fs(16)+' sans-serif';ctx.fillStyle='#889';
ctx.fillText('Transparent sphere | FTOP=center(0,0,0) | \u00b130 grid inside | Drag=rotate | Scroll=zoom',w/2,h-12);
ctx.fillText('5,184,000 sphered locations | (60\u00d760\u00d760)\u00d724 cycles | XY+XZ+YZ internal planes',w/2,h-32);

// === LEGEND moved to shared clickable #legend overlay ===

// === HIT/MISS STATS ===
ctx.textAlign='right';ctx.font='bold '+fs(18)+' monospace';
ctx.fillStyle='#44ff88';ctx.fillText('Hit:'+R.ch+'% | Miss:'+R.cmm+' AU',w-14,h-56);
ctx.fillStyle='#ff6644';ctx.fillText('Baseline:'+R.bh+'% | Miss:'+R.bmm+' AU',w-14,h-76);

// === HUD PANEL ===
const s3hud=document.getElementById('s3dHud');
if(s3hud){let sh='<div style="font:bold 11px sans-serif;color:#c8a96e;margin-bottom:6px;border-bottom:1px solid #5a4a32;padding-bottom:4px">\u25c9 3D SPHERE STATUS</div>';
sh+='<div style="display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font-size:9px">';
sh+='<span style="color:#889">FTOP</span><span style="color:#ffcc00">Center (0,0,0)</span>';
sh+='<span style="color:#889">Barrel</span><span style="color:#4488ff">'+bsph.addr+'</span>';
sh+='<span style="color:#889">Target</span><span style="color:#44ff88">'+tsph.addr+'</span>';
sh+='<span style="color:#889">Flight</span><span style="color:#ffd700">'+TF+' yr</span>';
sh+='<span style="color:#889">|v\u2080|</span><span style="color:#ffd700">'+Math.sqrt(V0[0]**2+V0[1]**2).toFixed(3)+' AU/yr</span>';
sh+='<span style="color:#889">Rotation</span><span style="color:#889">'+(s3rx*180/Math.PI).toFixed(1)+'\u00b0 / '+(s3ry*180/Math.PI).toFixed(1)+'\u00b0</span>';
sh+='<span style="color:#889">Zoom</span><span style="color:#889">'+s3z.toFixed(2)+'x</span>';
sh+='<span style="color:#889">Rs (\u221e)</span><span style="color:#ffd700">'+Rs+' AU = r=30</span>';
sh+='<span style="color:#889">Grid</span><span style="color:#ffd700">\u00b130 (XY+XZ+YZ planes)</span>';
sh+='<span style="color:#889">Gates</span><span style="color:#ffd700">12</span>';
sh+='</div>';
sh+='<div style="margin-top:6px;border-top:1px solid #3a3020;padding-top:4px;font-size:9px;color:#889">';
sh+='<b style="color:#c8a96e">GATE ADDRESSES:</b><br>';
for(let k=0;k<GT.length;k++){const gs=au2sph(GT[k][0],GT[k][1]);const isFib=SAL_FIB.has(k+1);
sh+='<span style="color:'+(isFib?'#ffd700':'#aa9')+'">G'+(k+1)+': '+gs.addr+' '+GI[k].spd.toFixed(1)+'</span><br>'}
sh+='</div>';
if(TGT_ORB){sh+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px">';
sh+='<b style="color:#44ff88">\u2609 TARGET ORBIT</b><br>';
sh+='v_orb: '+TGT_ORB.spd+' AU/yr<br>Period: '+TGT_ORB.period+' yr<br>';
const tSph=au2sph(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
sh+='At T_f: '+tSph.addr+'<br></div>'}
if(animT>0){sh+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px">';
const csph2=au2sph(getAnimPos()[0],getAnimPos()[1]);
sh+='<b style="color:#ffaa00">\u2604 COMET</b><br>'+csph2.addr+'<br>Cycle: '+csph2.time+'<br>';
sh+='T+'+animT.toFixed(3)+' yr | Gate '+getAnimGate()+'/12<br>';
sh+='Speed: '+getAnimSpd().toFixed(3)+' AU/yr</div>'}
if(ENER){sh+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px">';
sh+='<b style="color:#66aaff">\u26a1 ENERGY</b><br>';
sh+='|v\u2080|='+ENER.v0_kms+' km/s<br>dist='+ENER.dist_au+' AU<br>';
for(const[lbl,ed]of Object.entries(ENER.energy)){sh+=lbl+': '+ed.ke_j.toExponential(1)+' J<br>'}
sh+='</div>'}
s3hud.innerHTML=sh}

// === STORE FOR HOVER ===
c._gates=_s3dGates;
// === MOUSE CONTROLS ===
c.onmousedown=e=>{if(e.button===0){s3drg=true;lmx=e.clientX;lmy=e.clientY}};
c.onmouseup=()=>s3drg=false;
c.onmouseleave=()=>{s3drg=false;document.getElementById('s3dTip').style.display='none';document.getElementById('s3dCoord').textContent=''};
c.onmousemove=e=>{
if(s3drg){s3ry+=(e.clientX-lmx)*0.008;s3rx+=(e.clientY-lmy)*0.008;lmx=e.clientX;lmy=e.clientY;draw3DSphere();return}
const rc=c.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const stip=document.getElementById('s3dTip');const scoord=document.getElementById('s3dCoord');
const gates=c._gates||[];let found=null,bestD=900;
for(const g of gates){if(!g.vis)continue;const dx=mx-g.sx,dy=my-g.sy,d=dx*dx+dy*dy;if(d<bestD){bestD=d;found=g}}
if(found&&bestD<600){const g=found.g;const isFib=SAL_FIB.has(found.k+1);
const gsph=au2sph(GT[found.k][0],GT[found.k][1]);
stip.innerHTML='<b style="color:#ffd700">'+(isFib?'\u2605 Fib ':'')+'Gate '+(found.k+1)+'</b><br>'+
'<span style="color:#889">Sphere:</span> <span style="color:#ffd700">'+gsph.addr+'</span><br>'+
'<span style="color:#889">AU:</span> <span style="color:#ffd700">('+g.pos[0]+','+g.pos[1]+')</span><br>'+
'<span style="color:#889">Speed:</span> <span style="color:#ffd700">'+g.spd+' AU/yr</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+g.rsun+' AU</span><br>'+
'<span style="color:#889">Heading:</span> <span style="color:#ffd700">'+g.hdg+'\u00b0</span><br>'+
'<span style="color:#889">Curvature:</span> <span style="color:#ffd700">'+g.curv+' /AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+g.tg+' yr</span>';
stip.style.display='block';stip.style.left=Math.min(mx+16,rc.width-310)+'px';stip.style.top=Math.max(my-100,10)+'px';
scoord.textContent='Gate '+(found.k+1)+' | '+gsph.addr+' | '+g.spd+' AU/yr | T+'+g.tg+'yr'}
else{stip.style.display='none';scoord.textContent=''}};
c.onwheel=e=>{s3z*=e.deltaY>0?0.9:1.1;s3z=Math.max(0.3,Math.min(5,s3z));draw3DSphere();e.preventDefault()}}

// ═══ SALVO TAB — multi-barrel combined graph with focus rotation ═══
// SAL_FIB was originally declared here; it is now hoisted to the top of the
// script so the STATS page (and any other block that uses it before this line)
// does not hit a const TDZ ReferenceError.
// Rotate a 2D point around origin by angle (radians)
function rot2d(x,y,a){const c=Math.cos(a),s=Math.sin(a);return[x*c-y*s,x*s+y*c]}
// Build allBarrels array (shared between salvo + 3D)
function buildAllBarrels(){
const allB=[];
// origLbl keeps the barrel's identity label (PRIMARY / PARALLEL / SWARM@Nh)
// while lbl is the *role* label (PRIMARY for the active one, NON-PRIMARY otherwise).
// The active barrel is selected via salvoFocus (click in tab 7 or tab 8).
allB.push({ck:7,bp:B,v0:V0,nt:NT,gt:GT,col:'#ffd700',origLbl:'PRIMARY',lbl:'PRIMARY',miss:parseFloat(R.cmm),proj:(ENER&&ENER.primary_proj)||null});
if(PAR)allB.push({ck:'P',bp:PAR.bp,v0:PAR.v0,nt:PAR.nt,gt:PAR.gt,col:'#00dd88',origLbl:'PARALLEL (+'+PAR.off+'\u00b0)',lbl:'PARALLEL (+'+PAR.off+'\u00b0)',miss:PAR.miss,proj:PAR.proj||null});
for(let si=0;si<SWM.length;si++){const sw=SWM[si];allB.push({ck:sw.ck,bp:sw.bp,v0:sw.v0,nt:sw.nt,gt:sw.gt,col:SWM_COLS[si%SWM_COLS.length],origLbl:'SWARM @'+sw.ck+'h',lbl:'SWARM @'+sw.ck+'h',miss:sw.miss,proj:sw.proj||null})}
// Apply active-barrel role labels: the selected one is PRIMARY, others are NON-PRIMARY.
// salvoFocus===-1 means "default primary" \u2014 the first (index 0) keeps its native PRIMARY label.
const active=salvoFocus>=0&&salvoFocus<allB.length?salvoFocus:0;
for(let i=0;i<allB.length;i++){
  if(i===active){allB[i].isActive=true;allB[i].lbl='PRIMARY ('+allB[i].origLbl+')'}
  else{allB[i].isActive=false;allB[i].lbl='NON-PRIMARY ('+allB[i].origLbl+')'}
}
return allB}
// Compute rotation angle to put barrel bp at 7 o'clock (240°)
function focusAngle(bp){const curAng=Math.atan2(bp[1],bp[0]);const tgt7=(240)*Math.PI/180;return tgt7-curAng}

function drawSalvo(){
const c=document.getElementById('salC');const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
const cx=w/2,cy=h/2;
// Base scale + camera zoom (wheel) + pan (right-drag) give the salvo view
// the same world-coord camera model used by tab 1 SCOPE.
// Rs-normalised so an auto-grown scope still fits the viewport ("scroll out
// showing more AU" rather than "template spills off the canvas").
const baseSc=Math.min(w,h)/5.0*(RS_BASE/Math.max(Rs,1e-9));const sc=baseSc*salCamZ;
ctx.fillStyle='#080810';ctx.fillRect(0,0,w,h);

const allB=buildAllBarrels();
// Focus rotation: if a barrel is focused, rotate everything so that barrel sits at 7 o'clock
const fIdx=salvoFocus>=0&&salvoFocus<allB.length?salvoFocus:-1;
const rotA=fIdx>=0?focusAngle(allB[fIdx].bp):0;
// t2: world AU → screen pixels, applying (rotate → pan → scale → center).
const t2=(x,y)=>{const[rx2,ry2]=rot2d(x-salCamX,y-salCamY,rotA);return[cx+rx2*sc,cy-ry2*sc]};
// inverse: screen → world AU (used by wheel zoom-to-cursor)
const s2w=(sx,sy)=>{const rx2=(sx-cx)/sc,ry2=-(sy-cy)/sc;const[ix,iy]=rot2d(rx2,ry2,-rotA);return[ix+salCamX,iy+salCamY]};

// Transparent 3D Flower-of-Life sphere grid — projected flat (z ignored) for salvo 2D view.
// This is the 3D expansion of the 2D scope template, drawn as a transparent grid that
// fills the space around every barrel + target.
if(DV.X){const to3d=(x,y,z)=>t2(x,y);
drawFoL3DSpheres(ctx,to3d,{alpha:0.18,use3D:false})}
// Reference orbit rings
for(const[rr,col] of [[0.72,'rgba(210,180,100,.15)'],[1.0,'rgba(120,170,255,.2)'],[1.52,'rgba(255,120,90,.15)'],[Rs,'rgba(220,185,120,.2)']]) {
ctx.beginPath();for(let i=0;i<=64;i++){const a=i*Math.PI*2/64;const[px,py]=t2(rr*Math.cos(a),rr*Math.sin(a));if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
ctx.strokeStyle=col;ctx.lineWidth=1;ctx.stroke()}
// Solar system (rotated along with salvo focus).
c._solarHits=drawSolarSystem2D(ctx,t2,{scale:sc*0.8});
// FTOP (orientation anchor, NOT a gravitating mass)
const[ox,oy]=t2(0,0);ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=12;
ctx.beginPath();ctx.arc(ox,oy,6,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
ctx.font='bold '+fs(28)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';ctx.fillText('FTOP',ox,oy-14);
// Clock numbers on scope ring (rotated)
ctx.font='bold '+fs(32)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
for(let i=1;i<=12;i++){const a=(90-i*30)*Math.PI/180;
const[cx2,cy2]=t2((Rs+0.18)*Math.cos(a),(Rs+0.18)*Math.sin(a));
ctx.fillStyle=i===7?'#66aaff':'rgba(160,144,112,.5)';ctx.fillText(i,cx2,cy2)}

// TARGET (common for all)
const[tgsx,tgsy]=t2(TG[0],TG[1]);
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=18;
ctx.beginPath();ctx.arc(tgsx,tgsy,14,0,Math.PI*2);ctx.fillStyle='#44ff88';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tgsx,tgsy,14,0,Math.PI*2);ctx.stroke();
ctx.strokeStyle='rgba(68,255,136,.3)';ctx.lineWidth=1;
ctx.beginPath();ctx.moveTo(tgsx-20,tgsy);ctx.lineTo(tgsx+20,tgsy);ctx.stroke();
ctx.beginPath();ctx.moveTo(tgsx,tgsy-20);ctx.lineTo(tgsx,tgsy+20);ctx.stroke();
ctx.font='bold '+fs(32)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';
ctx.fillText('TARGET X',tgsx,tgsy-26);
ctx.font=fs(24)+' monospace';ctx.fillText('('+TG[0].toFixed(2)+', '+TG[1].toFixed(2)+') AU',tgsx,tgsy+32);

// Store barrel screen positions for click detection
const salvoHits=[];

// Draw all trajectories with Fibonacci gate numbering
for(let bi=0;bi<allB.length;bi++){const b=allB[bi];const isFocused=(bi===fIdx);const isPrimary=!!b.isActive;
const bright=isFocused||fIdx<0;
ctx.beginPath();ctx.strokeStyle=b.col+(bright?'':'55');ctx.lineWidth=(isFocused||isPrimary)?3:1.5;
if(!isPrimary&&!isFocused)ctx.setLineDash([5,4]);
for(let i=0;i<b.nt.length;i++){const[sx,sy]=t2(b.nt[i][0],b.nt[i][1]);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([]);
// Barrel marker
const[bsx,bsy]=t2(b.bp[0],b.bp[1]);
const bR=(isFocused||isPrimary)?8:5;
ctx.beginPath();ctx.arc(bsx,bsy,bR,0,Math.PI*2);ctx.fillStyle=b.col;ctx.fill();
ctx.strokeStyle=isFocused?'#fff':'rgba(255,255,255,.5)';ctx.lineWidth=isFocused?2:1;ctx.beginPath();ctx.arc(bsx,bsy,bR,0,Math.PI*2);ctx.stroke();
// Focus ring
if(isFocused){ctx.beginPath();ctx.arc(bsx,bsy,bR+5,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([])}
ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle=b.col;ctx.textAlign='center';
ctx.fillText(b.lbl+(isFocused?' \u2190FOCUS':''),bsx,bsy-16);
ctx.font=fs(20)+' monospace';ctx.fillText('('+b.bp[0].toFixed(2)+','+b.bp[1].toFixed(2)+')',bsx,bsy+20);
salvoHits.push({x:bsx,y:bsy,r:bR+8,idx:bi});
// Convergence line
ctx.beginPath();ctx.setLineDash([3,6]);ctx.strokeStyle=b.col+(bright?'33':'15');ctx.lineWidth=0.8;
ctx.moveTo(bsx,bsy);ctx.lineTo(tgsx,tgsy);ctx.stroke();ctx.setLineDash([]);
// Gate dots with Fibonacci ratio numbering
for(let k=0;k<b.gt.length;k++){const[gsx,gsy]=t2(b.gt[k][0],b.gt[k][1]);
const gR=(isFocused||isPrimary)?4:2.5;const isFib=SAL_FIB.has(k+1);
ctx.beginPath();ctx.arc(gsx,gsy,gR,0,Math.PI*2);ctx.fillStyle=b.col+(bright?'cc':'55');ctx.fill();
if(isFib&&bright){
// Fibonacci ring + number label
ctx.save();ctx.shadowColor=b.col;ctx.shadowBlur=8;
ctx.beginPath();ctx.arc(gsx,gsy,gR+4,0,Math.PI*2);ctx.strokeStyle=b.col;ctx.lineWidth=1.5;ctx.stroke();ctx.restore();
ctx.font='bold '+fs(24)+' monospace';ctx.fillStyle=b.col;ctx.textAlign='center';ctx.fillText('F'+(k+1),gsx,gsy-gR-10)}
else if(bright&&(isPrimary||isFocused)){
ctx.font=fs(20)+' monospace';ctx.fillStyle=b.col+'aa';ctx.textAlign='center';ctx.fillText(k+1,gsx,gsy-gR-8)}
// Velocity direction arrows at gates (for focused/primary)
if(bright&&(isPrimary||isFocused)){
const gData2=GI[k];
const vAng2=Math.atan2(-gData2.vy,gData2.vx);const vLen2=10;
const vEx2=gsx+vLen2*Math.cos(vAng2),vEy2=gsy+vLen2*Math.sin(vAng2);
ctx.beginPath();ctx.moveTo(gsx,gsy);ctx.lineTo(vEx2,vEy2);ctx.strokeStyle=b.col+'99';ctx.lineWidth=1.5;ctx.stroke();
const vAh2=3;ctx.beginPath();ctx.moveTo(vEx2,vEy2);ctx.lineTo(vEx2-vAh2*Math.cos(vAng2-.4),vEy2-vAh2*Math.sin(vAng2-.4));
ctx.lineTo(vEx2-vAh2*Math.cos(vAng2+.4),vEy2-vAh2*Math.sin(vAng2+.4));ctx.closePath();ctx.fillStyle=b.col+'99';ctx.fill()}}
// Midpoint annotations for focused/primary barrel
if(bright&&(isPrimary||isFocused)){
for(let mk2=0;mk2<b.gt.length;mk2++){
const mgD=GI[mk2];const prevS=mk2===0?b.bp:b.gt[mk2-1];
const smx=(prevS[0]+b.gt[mk2][0])/2,smy=(prevS[1]+b.gt[mk2][1])/2;
const[smpx,smpy]=t2(smx,smy);
ctx.font=fs(12)+' sans-serif';ctx.textAlign='center';
ctx.fillStyle='rgba(220,200,140,.45)';ctx.fillText('\u2220'+Math.abs(mgD.dhdg).toFixed(1)+'\u00b0',smpx,smpy-8);
ctx.fillStyle=mgD.dspd>=0?'rgba(100,255,150,.4)':'rgba(255,130,100,.4)';
ctx.fillText((mgD.dspd>=0?'+':'')+mgD.dspd.toFixed(2),smpx,smpy+6)}}}

// Tensor mapping lines — scope gate → trajectory gate (rotated)
ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(255,215,140,.40)';ctx.lineWidth=1.1;
for(let km2=0;km2<Math.min(GXY.length,GT.length);km2++){
const[smx1,smy1]=t2(GXY[km2][0],GXY[km2][1]);const[smx2,smy2]=t2(GT[km2][0],GT[km2][1]);
ctx.beginPath();ctx.moveTo(smx1,smy1);ctx.lineTo(smx2,smy2);ctx.stroke()}ctx.setLineDash([]);
// Clock-face gate dots on scope ring (rotated)
for(let sg=0;sg<GXY.length;sg++){const[sgx3,sgy3]=t2(GXY[sg][0],GXY[sg][1]);
ctx.beginPath();ctx.arc(sgx3,sgy3,3,0,Math.PI*2);ctx.fillStyle='rgba(220,185,120,.25)';ctx.fill()}

// Target orbital path (rotated)
if(TGT_ORB){
if(TGT_ORB.history){ctx.beginPath();ctx.strokeStyle='rgba(255,140,60,.12)';ctx.lineWidth=1;ctx.setLineDash([3,5]);
const shst=Math.max(1,Math.floor(TGT_ORB.history.length/150));
for(let ih2=0;ih2<TGT_ORB.history.length;ih2+=shst){const[shx,shy]=t2(TGT_ORB.history[ih2][0],TGT_ORB.history[ih2][1]);
if(ih2===0)ctx.moveTo(shx,shy);else ctx.lineTo(shx,shy)}ctx.stroke();ctx.setLineDash([])}
if(TGT_ORB.path){ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.12)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
const sfst=Math.max(1,Math.floor(TGT_ORB.path.length/150));
for(let ip2=0;ip2<TGT_ORB.path.length;ip2+=sfst){const[sfx,sfy]=t2(TGT_ORB.path[ip2][0],TGT_ORB.path[ip2][1]);
if(ip2===0)ctx.moveTo(sfx,sfy);else ctx.lineTo(sfx,sfy)}ctx.stroke();ctx.setLineDash([]);
// Target at T_f
const[stfx,stfy]=t2(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
ctx.beginPath();ctx.arc(stfx,stfy,4,0,Math.PI*2);ctx.strokeStyle='rgba(68,255,136,.3)';ctx.lineWidth=1;ctx.setLineDash([2,2]);ctx.stroke();ctx.setLineDash([]);
ctx.font=fs(12)+' monospace';ctx.fillStyle='rgba(68,255,136,.4)';ctx.textAlign='center';ctx.fillText('T@Tf',stfx,stfy+12)}
// Target details
ctx.font=fs(18)+' monospace';ctx.fillStyle='#88ccaa';ctx.textAlign='center';
ctx.fillText('v_orb='+TGT_ORB.spd+' AU/yr P='+TGT_ORB.period+' yr',tgsx,tgsy+52);
ctx.fillStyle='rgba(136,102,102,.5)';ctx.fillText('\u03b2='+TGT_ORB.beta.toExponential(2)+' \u03b3='+TGT_ORB.gamma.toFixed(6),tgsx,tgsy+70);
// Hit radius ring around target
ctx.beginPath();ctx.arc(tgsx,tgsy,HR*sc,0,Math.PI*2);ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1;ctx.setLineDash([3,5]);ctx.stroke();ctx.setLineDash([])}

// Compute interpolated position for each barrel at current animT
const salvoPositions=[];
for(let bi=0;bi<allB.length;bi++){const b=allB[bi];
const f2=TF>0?Math.max(0,Math.min(1,animT/TF)):0;const nPts2=b.nt.length-1;
const idx2=f2*nPts2;const i0b=Math.floor(idx2),i1b=Math.min(i0b+1,nPts2);const fr2=idx2-i0b;
const px2=b.nt[i0b][0]+(b.nt[i1b][0]-b.nt[i0b][0])*fr2;
const py2=b.nt[i0b][1]+(b.nt[i1b][1]-b.nt[i0b][1])*fr2;
const dTgt2=Math.sqrt((px2-TG[0])**2+(py2-TG[1])**2);
const gk2=Math.min(Math.floor(animT/DTG),11);
salvoPositions.push({lbl:b.lbl,col:b.col,x:px2,y:py2,dTgt:dTgt2,gate:gk2+1,isFired:playing&&b.isActive})}

// Draw animated comets for ALL barrels
for(let bi=0;bi<allB.length;bi++){const b=allB[bi];const sp=salvoPositions[bi];
if(animT<=0)continue;
const[asx2,asy2]=t2(sp.x,sp.y);const isFired=sp.isFired;
const cR=isFired?10:5;
ctx.save();ctx.shadowColor=isFired?'#ff8800':b.col;ctx.shadowBlur=isFired?25:10;
ctx.beginPath();ctx.arc(asx2,asy2,cR,0,Math.PI*2);ctx.fillStyle=isFired?'#ffaa00':b.col;ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(asx2,asy2,cR,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=isFired?2:1;ctx.stroke();
if(isFired){ctx.font='bold '+fs(28)+' monospace';ctx.fillStyle='#ffaa00';ctx.textAlign='left';
ctx.fillText('T+'+animT.toFixed(3)+' yr',asx2+14,asy2+6);
ctx.font='bold '+fs(22)+' monospace';ctx.fillStyle='#fff';ctx.fillText('\u25b6 FIRED: '+b.lbl,asx2+14,asy2+32)}}

// Title
const firedLbl=playing&&fIdx>=0?' | \u25b6 FIRING: '+allB[fIdx].lbl:playing?' | \u25b6 FIRING: PRIMARY':'';
ctx.font='bold '+fs(34)+' "Times New Roman",serif';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
ctx.fillText('SALVO \u2014 Multi-Barrel Convergence Graph',w/2,34);
ctx.font=fs(24)+' sans-serif';ctx.fillStyle='#889';
const focLbl=fIdx>=0?' | FOCUS: '+allB[fIdx].lbl+' \u2192 7h':' | Click barrel to focus';
ctx.fillText(allB.length+' barrels \u2192 Target X'+focLbl+firedLbl+' | SPACE=play | P=par S=swm',w/2,60);

// Legend — bottom-left (matching Scope)
// Legend moved to shared clickable #legend overlay (bottom-left fixed panel).
let llyS=h-196;

// AU ruler (bottom-left) — nice round bar whose length reflects current zoom
{const targetPx=140;const auForBar=targetPx/sc;
const niceAU=(function(a){const pow=Math.pow(10,Math.floor(Math.log10(a)));const n=a/pow;
let m=1;if(n>=5)m=5;else if(n>=2)m=2;return m*pow})(auForBar);
const barPx=niceAU*sc;
ctx.save();ctx.strokeStyle='#c8a96e';ctx.lineWidth=2;
ctx.beginPath();ctx.moveTo(16,h-16);ctx.lineTo(16+barPx,h-16);
ctx.moveTo(16,h-22);ctx.lineTo(16,h-10);
ctx.moveTo(16+barPx,h-22);ctx.lineTo(16+barPx,h-10);ctx.stroke();
ctx.font='bold '+fs(14)+' monospace';ctx.fillStyle='#c8a96e';ctx.textAlign='left';
const rulerLbl=niceAU>=1?niceAU.toFixed(niceAU>=10?0:1)+' AU':(niceAU*1.496e8).toFixed(0)+' km';
ctx.fillText(rulerLbl,20,h-26);
ctx.restore()}
// Hit/miss stats — bottom-right canvas
ctx.textAlign='right';ctx.font='bold '+fs(20)+' monospace';
ctx.fillStyle='#44ff88';ctx.fillText('Hit:'+R.ch+'% | Miss:'+R.cmm+' AU',w-14,h-14);
ctx.fillStyle='#ff6644';ctx.fillText('Baseline:'+R.bh+'% | Miss:'+R.bmm+' AU',w-14,h-38);
const gapS=Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2);
ctx.fillStyle='#667';ctx.font=fs(16)+' sans-serif';
ctx.fillText('Gap: '+gapS.toFixed(4)+' AU | |v\u2080|='+Math.sqrt(V0[0]**2+V0[1]**2).toFixed(3)+' AU/yr | Zoom='+salCamZ.toFixed(salCamZ<1?3:2)+'\u00d7 | RClick=pan \u00b7 Wheel=zoom \u00b7 Click=focus \u00b7 Hover=data',w-14,h-56);

// Info panel (right side) \u2014 dense multi-section readout + zoom controls
// Totals across the whole barrel array (computed before per-barrel rows so the
// summary sits at the top where the eye lands first).
let sumKE=0,sumP=0,sumDv=0,sumFuel=0,sumMass=0,hitCount=0,missCount=0,viewR=0;
for(const b of allB){if(b.proj){sumKE+=b.proj.ke_j;sumP+=b.proj.power_avg_W;sumDv+=b.proj.dv_total_kms;sumFuel+=b.proj.fuel_frac;sumMass+=b.proj.mass_kg}
if(b.miss<HR)hitCount++;else missCount++;
const rb=Math.sqrt(b.bp[0]**2+b.bp[1]**2);if(rb>viewR)viewR=rb}
const pHit=0.995;const pCombined=1-Math.pow(1-pHit,allB.length);
const auPerPx=1/sc;const aui=auPerPx*Math.min(c.width,c.height);
let si2='<div style="font:bold 11px sans-serif;color:#c8a96e;margin-bottom:4px;letter-spacing:1px">\u25c6 SALVO STATUS</div>';
// Zoom controls row
si2+='<div style="display:flex;gap:4px;align-items:center;margin-bottom:6px;font-size:9px;color:#889">';
si2+='<span>Zoom</span>';
si2+='<button onclick="salCamZ=Math.max(MIN_ZOOM,salCamZ/1.3);draw()" style="font:bold 10px monospace;padding:1px 6px;border:1px solid #5a4a32;background:#1a1610;color:#c8a96e;border-radius:3px;cursor:pointer">\u2212</button>';
si2+='<span style="color:#ffd700;font-weight:bold;min-width:40px;text-align:center">'+salCamZ.toFixed(salCamZ<1?3:2)+'\u00d7</span>';
si2+='<button onclick="salCamZ=Math.min(MAX_ZOOM,salCamZ*1.3);draw()" style="font:bold 10px monospace;padding:1px 6px;border:1px solid #5a4a32;background:#1a1610;color:#c8a96e;border-radius:3px;cursor:pointer">+</button>';
si2+='<button onclick="salCamZ=1;salCamX=0;salCamY=0;draw()" style="margin-left:4px;font:bold 9px monospace;padding:1px 6px;border:1px solid #5a4a32;background:#1a1610;color:#c8a96e;border-radius:3px;cursor:pointer">RESET</button>';
si2+='</div>';
if(fIdx>=0){si2+='<div style="margin-bottom:6px;padding:4px;background:rgba(255,215,0,.1);border:1px solid #ffd700;border-radius:3px;font-size:9px;color:#ffd700">\u26a0 FOCUS: <b>'+allB[fIdx].origLbl+'</b> \u2192 7 o\'clock<br>All others shifted in formation relative</div>'}
si2+='<div style="margin-bottom:6px;font-size:9px;color:#889">All '+allB.length+' barrels converging on TARGET ('+TG[0].toFixed(2)+', '+TG[1].toFixed(2)+') AU \u00b7 FOV \u2248 '+aui.toFixed(3)+' AU</div>';
// Per-barrel cards \u2014 now denser: gap, angle to target, approach angle,
// clock sector, projectile+energy, and an angular indicator.
for(let bi=0;bi<allB.length;bi++){const b=allB[bi];const vm=Math.sqrt(b.v0[0]**2+b.v0[1]**2);const ok=b.miss<0.1;const isFoc=bi===fIdx;
const rb=Math.sqrt(b.bp[0]**2+b.bp[1]**2);
const abAng=((Math.atan2(b.bp[1],b.bp[0])*180/Math.PI)%360+360)%360;
const clock=((Math.round(((90-abAng+360)%360)/30))||12);
const dxTgt=TG[0]-b.bp[0],dyTgt=TG[1]-b.bp[1];const rngT=Math.sqrt(dxTgt*dxTgt+dyTgt*dyTgt);
const apprAng=((Math.atan2(dyTgt,dxTgt)*180/Math.PI)%360+360)%360;
const vAng=((Math.atan2(b.v0[1],b.v0[0])*180/Math.PI)%360+360)%360;
const leadAng=((vAng-apprAng+540)%360)-180; // signed lead angle vs. straight line to target
si2+='<div style="border-left:3px solid '+b.col+';padding:4px 6px;margin:4px 0;font-size:9px;cursor:pointer;'+(isFoc?'background:rgba(255,255,255,.06);border:1px solid '+b.col+';border-radius:3px':'')+'" onclick="salvoFocus='+(isFoc?-1:bi)+';draw()">';
si2+='<b style="color:'+b.col+'">'+(b.isActive?'\u2605 ':'')+b.origLbl+'</b>'+(b.isActive?' <span style="color:#fff">[PRIMARY]</span>':'');
if(isFoc)si2+=' <span style="color:#fff">(FOCUSED \u2192 7h)</span>';
si2+='<br><span style="color:#889">Pos</span> ('+b.bp[0].toFixed(3)+', '+b.bp[1].toFixed(3)+') AU';
si2+=' <span style="color:#889">R</span> '+rb.toFixed(3)+' <span style="color:#889">@</span> '+abAng.toFixed(1)+'\u00b0';
si2+=' <span style="color:#889">\u2248</span> '+clock+'h';
si2+='<br><span style="color:#889">|v\u2080|</span> '+vm.toFixed(3)+' AU/yr';
si2+=' <span style="color:#889">Hdg</span> '+vAng.toFixed(1)+'\u00b0';
si2+=' <span style="color:#889">Lead</span> '+(leadAng>=0?'+':'')+leadAng.toFixed(1)+'\u00b0';
si2+='<br><span style="color:#889">Range\u2192T</span> '+rngT.toFixed(3)+' AU';
si2+=' <span style="color:#889">Miss</span> <span style="color:'+(ok?'#44ff88':'#ff4444')+'">'+b.miss.toExponential(2)+' '+(ok?'\u2714':'\u2718')+'</span>';
if(b.proj){const pr=b.proj;si2+='<br><span style="color:#ffd700">\u2300'+pr.diameter_m+' m \u00b7 m='+pr.mass_kg.toExponential(2)+' kg</span>';
si2+='<br><span style="color:#889">KE</span> '+pr.ke_j.toExponential(2)+' J ('+pr.tnt_kt.toFixed(2)+' kt)';
si2+=' <span style="color:#889">P<sub>avg</sub></span> '+pr.power_avg_W.toExponential(2)+' W';
si2+='<br><span style="color:#889">\u0394v</span> '+pr.dv_total_kms.toFixed(2)+' km/s';
si2+=' <span style="color:#889">Fuel</span> '+(pr.fuel_frac*100).toFixed(2)+'%';
si2+=' <span style="color:#889">Isp</span> '+pr.Isp_s+'s'}
si2+='</div>'}
// Array totals + combined probability (bottom summary)
si2+='<div style="border-top:1px solid #5a4a32;margin-top:8px;padding-top:6px;font:10px monospace;color:#c8a96e">';
si2+='<div style="color:#c8a96e;font-weight:bold;margin-bottom:3px">ARRAY TOTALS</div>';
si2+='<span style="color:#889">Barrels</span> '+allB.length+' <span style="color:#889">\u00b7 Hits</span> <span style="color:#44ff88">'+hitCount+'</span> <span style="color:#889">Misses</span> <span style="color:#ff6644">'+missCount+'</span><br>';
si2+='<span style="color:#889">\u03a3 mass</span> '+sumMass.toExponential(2)+' kg<br>';
si2+='<span style="color:#889">\u03a3 KE</span> '+sumKE.toExponential(2)+' J<br>';
si2+='<span style="color:#889">\u03a3 P<sub>avg</sub></span> <span style="color:#ffaa44">'+sumP.toExponential(2)+' W</span><br>';
si2+='<span style="color:#889">\u03a3 \u0394v</span> '+sumDv.toFixed(2)+' km/s \u00b7 <span style="color:#889">avg fuel</span> '+(100*sumFuel/Math.max(1,allB.length)).toFixed(2)+'%<br>';
si2+='<span style="color:#889">P(any hit)</span> <span style="color:#44ff88;font-weight:bold">'+(pCombined*100).toFixed(6)+'%</span>';
si2+=' <span style="color:#667;font-size:8px">[1 \u2212 \u220f(1 \u2212 p\u1d62)]</span>';
si2+='</div>';
si2+='<div style="margin-top:6px;font:8px sans-serif;color:#556;line-height:1.4">RClick+drag = pan \u00b7 wheel = zoom to cursor \u00b7 click barrel = focus/unfocus at 7h \u00b7 \u2605 = active PRIMARY across tabs</div>';
document.getElementById('salInfo').innerHTML=si2;
document.getElementById('salInfo').style.pointerEvents='auto';

// ═══ PERSISTENT LIVE TRACKING PANEL (bottom-left, always visible) ═══
let lv='<div style="font:bold 12px sans-serif;color:#ffd700;margin-bottom:6px;border-bottom:1px solid #5a4a32;padding-bottom:4px">\u25c9 LIVE PROJECTILE TRACKING \u2014 T+'+animT.toFixed(3)+' yr</div>';
lv+='<table style="width:100%;border-collapse:collapse;font-size:9px">';
lv+='<tr style="color:#889;border-bottom:1px solid #3a3020"><th style="text-align:left;padding:2px 4px">Barrel</th><th>Position</th><th>D(TGT)</th><th>Gate</th><th>Status</th></tr>';
for(let bi=0;bi<salvoPositions.length;bi++){const sp=salvoPositions[bi];
lv+='<tr style="border-bottom:1px solid rgba(90,74,50,.15);color:'+sp.col+'">';
lv+='<td style="padding:2px 4px;font-weight:bold">'+sp.lbl+'</td>';
lv+='<td style="text-align:center">('+sp.x.toFixed(3)+', '+sp.y.toFixed(3)+')</td>';
lv+='<td style="text-align:center;color:'+(sp.dTgt<0.1?'#44ff88':'#c8a96e')+'">'+sp.dTgt.toFixed(4)+'</td>';
lv+='<td style="text-align:center">'+sp.gate+'/12</td>';
lv+='<td style="text-align:center">'+(sp.isFired?'<span style="color:#ff8800">\u25b6 FIRED</span>':animT>0?'\u2192 In flight':'\u23f8 Ready')+'</td>';
lv+='</tr>'}
lv+='</table>';
if(animT>0){const closest=salvoPositions.reduce((a,b)=>a.dTgt<b.dTgt?a:b);
lv+='<div style="margin-top:4px;font:bold 10px monospace;color:#44ff88">Nearest: '+closest.lbl+' \u2192 '+closest.dTgt.toFixed(4)+' AU from target</div>'}
document.getElementById('salLive').innerHTML=lv;

// Store trajectory screen points for hover detection
c._salvoHits=salvoHits;
c._salvoPaths=[];
for(let bi=0;bi<allB.length;bi++){const b=allB[bi];const pts=[];
const step=Math.max(1,Math.floor(b.nt.length/100));
for(let i=0;i<b.nt.length;i+=step){const[sx,sy]=t2(b.nt[i][0],b.nt[i][1]);
pts.push({sx:sx,sy:sy,wx:b.nt[i][0],wy:b.nt[i][1],t:i/(b.nt.length-1)*TF,bi:bi})}
c._salvoPaths.push({lbl:b.lbl,col:b.col,pts:pts})}}

// Salvo canvas click — select barrel
document.getElementById('salC').addEventListener('click',e=>{
const c2=document.getElementById('salC');const rc=c2.getBoundingClientRect();
const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const hits=c2._salvoHits||[];let best=-1,bestD=99999;
for(const h of hits){const dx=mx-h.x,dy=my-h.y,d=dx*dx+dy*dy;if(d<h.r*h.r&&d<bestD){bestD=d;best=h.idx}}
if(best>=0){salvoFocus=salvoFocus===best?-1:best;draw()}});

// Salvo canvas hover — trajectory path tooltip
document.getElementById('salC').addEventListener('mousemove',e=>{
const c2=document.getElementById('salC');const rc=c2.getBoundingClientRect();
const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const stip=document.getElementById('salTip');
const paths=c2._salvoPaths||[];
let found=null,bestD2=900; // 30px threshold squared
for(const p of paths){for(const pt of p.pts){
const dx=mx-pt.sx,dy=my-pt.sy,d2=dx*dx+dy*dy;
if(d2<bestD2){bestD2=d2;found={lbl:p.lbl,col:p.col,wx:pt.wx,wy:pt.wy,t:pt.t,bi:pt.bi}}}}
if(found){const rr=Math.sqrt(found.wx*found.wx+found.wy*found.wy);
const dTgt=Math.sqrt((found.wx-TG[0])**2+(found.wy-TG[1])**2);
const gk2=Math.min(Math.floor(found.t/DTG),11);
stip.innerHTML='<b style="color:'+found.col+'">'+found.lbl+'</b><br>'+
'<span style="color:#889">Position:</span> <span style="color:#ffd700">('+found.wx.toFixed(4)+', '+found.wy.toFixed(4)+') AU</span><br>'+
'<span style="color:#889">R(FTOP):</span> <span style="color:#ffd700">'+rr.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">D(Target):</span> <span style="color:'+(dTgt<0.1?'#44ff88':'#ffd700')+'">'+dTgt.toFixed(4)+' AU</span><br>'+
'<span style="color:#889">Time:</span> <span style="color:#ffd700">T+'+found.t.toFixed(3)+' yr</span><br>'+
'<span style="color:#889">Gate:</span> <span style="color:#ffd700">'+(gk2+1)+'/12</span>';
stip.style.display='block';stip.style.left=Math.min(mx+16,rc.width-330)+'px';stip.style.top=Math.max(my-80,10)+'px'}
else{stip.style.display='none'}});
document.getElementById('salC').addEventListener('mouseleave',()=>{document.getElementById('salTip').style.display='none';salPanDrg=false});

// === SALVO pan + zoom (world-coord camera, same idiom as tab 1) ===
{const sC2=document.getElementById('salC');if(sC2&&!sC2._panWired){sC2._panWired=true;
sC2.addEventListener('contextmenu',e=>e.preventDefault());
sC2.addEventListener('mousedown',e=>{if(e.button===2){salPanDrg=true;salLmx=e.clientX;salLmy=e.clientY;e.preventDefault()}});
sC2.addEventListener('mouseup',e=>{if(e.button===2)salPanDrg=false});
sC2.addEventListener('mousemove',e=>{if(!salPanDrg)return;
// Pan speed scales with zoom (so the canvas tracks the cursor 1:1)
const baseSc=Math.min(sC2.width||800,sC2.height||600)/5.0*(RS_BASE/Math.max(Rs,1e-9));const sc=baseSc*salCamZ;
const dx=e.clientX-salLmx,dy=e.clientY-salLmy;
// Rotation of the salvo view is compensated by focusAngle; pan in world AU,
// un-rotating the screen delta so the drag feels natural regardless of focus rotation.
const fIdx2=salvoFocus>=0&&salvoFocus<buildAllBarrels().length?salvoFocus:-1;
const rotA2=fIdx2>=0?focusAngle(buildAllBarrels()[fIdx2].bp):0;
const rdx=dx*Math.cos(-rotA2)+(-dy)*Math.sin(-rotA2);
const rdy=-dx*Math.sin(-rotA2)+(-dy)*Math.cos(-rotA2);
salCamX-=rdx/sc;salCamY-=rdy/sc;
salLmx=e.clientX;salLmy=e.clientY;drawSalvo()});
sC2.addEventListener('wheel',e=>{e.preventDefault();
const rc=sC2.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const baseSc=Math.min(sC2.width||800,sC2.height||600)/5.0*(RS_BASE/Math.max(Rs,1e-9));
const oldZ=salCamZ;const oldSc=baseSc*oldZ;
const cxP=sC2.width/2,cyP=sC2.height/2;
// World point under cursor before zoom (accounting for rotation)
const fIdx2=salvoFocus>=0&&salvoFocus<buildAllBarrels().length?salvoFocus:-1;
const rotA2=fIdx2>=0?focusAngle(buildAllBarrels()[fIdx2].bp):0;
const rx=(mx-cxP)/oldSc,ry=-(my-cyP)/oldSc;
const[ix,iy]=rot2d(rx,ry,-rotA2);const wx=ix+salCamX,wy=iy+salCamY;
salCamZ*=e.deltaY>0?0.88:1.14;salCamZ=Math.max(MIN_ZOOM,Math.min(MAX_ZOOM,salCamZ));
const newSc=baseSc*salCamZ;
const rx2=(mx-cxP)/newSc,ry2=-(my-cyP)/newSc;
const[ix2,iy2]=rot2d(rx2,ry2,-rotA2);
salCamX=wx-ix2;salCamY=wy-iy2;
drawSalvo()},{passive:false})}}

// ═══ SPHERE NAV — Dimensional Sphere Coordinate System ═══
// Mapping: scope radius Rs = infinity boundary = grid ±30
// Address format: X°,xz,xx,xy  where X° is polar angle 0-360
// Grid: 60 divisions per axis (-30..+30), 24 looped time cycles
// Total addressable: (60×60×60)×24 = 5,184,000 sphere locations
let sphZoom=1,sphPanX=0,sphPanY=0,sphDrg=false,sphLmx=0,sphLmy=0;

// Legacy 2D sphere-address helper used by tabs 9 + 10. Thin wrapper over
// sphereAddr() so the dual-angle (phi, theta) + 3-axis grid format is
// produced everywhere consistently. Accepts an optional z-component.
function au2sph(ax,ay,az){return sphereAddr(ax,ay,az)}

function drawSphere(){
const c=document.getElementById('sphC');const ctx=c.getContext('2d');
const w=c.width=c.parentElement.clientWidth;const h=c.height=c.parentElement.clientHeight;
const cx=w/2,cy=h/2;
const scopeR=Math.min(w,h)/2.3*sphZoom;
// Transform: grid coord (-30..30) → screen pixel
const g2s=(gx,gy)=>[cx+(gx/30-sphPanX)*scopeR,cy-(gy/30-sphPanY)*scopeR];
// Inverse: screen → grid
const s2g=(sx,sy)=>[(sx-cx)/scopeR*30+sphPanX*30,(cy-sy)/scopeR*30+sphPanY*30];

// Background
const bg=ctx.createRadialGradient(cx,cy,0,cx,cy,scopeR);
bg.addColorStop(0,'#0c0c1e');bg.addColorStop(0.85,'#080814');bg.addColorStop(1,'#040408');
ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);

// Transparent 2D Flower-of-Life sphere grid — the nav-ball depiction.
// AU coords map to grid coords via /Rs*30; route through g2s so the
// 3D sphere lattice appears flattened as the 2D nav-ball version.
if(DV.X){const to3d=(x,y,z)=>g2s(x/Rs*30,y/Rs*30);
drawFoL3DSpheres(ctx,to3d,{alpha:0.15,use3D:false})}

// ═══ SCOPE BOUNDARY CIRCLE (= INFINITY) ═══
ctx.save();ctx.shadowColor='#c8a96e';ctx.shadowBlur=20;
ctx.beginPath();ctx.arc(cx-sphPanX*scopeR,cy+sphPanY*scopeR,scopeR,0,Math.PI*2);
ctx.strokeStyle='rgba(200,169,110,.5)';ctx.lineWidth=3;ctx.stroke();ctx.restore();
// Infinity label
ctx.font='bold '+fs(28)+' "Times New Roman",serif';ctx.fillStyle='rgba(200,169,110,.6)';ctx.textAlign='center';
ctx.fillText('\u221e = '+Rs.toFixed(2)+' AU (Scope Boundary)',cx,cy+sphPanY*scopeR+scopeR+30);

// ═══ GRID LINES (-30 to 30) ═══
if(DV.Grid){ctx.strokeStyle='rgba(80,70,55,.18)';ctx.lineWidth=0.5;
for(let g=-30;g<=30;g+=5){
// Vertical line (constant x)
const[x1,y1]=g2s(g,-30);const[x2,y2]=g2s(g,30);
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
// Horizontal line (constant y)
const[x3,y3]=g2s(-30,g);const[x4,y4]=g2s(30,g);
ctx.beginPath();ctx.moveTo(x3,y3);ctx.lineTo(x4,y4);ctx.stroke()}
// Finer sub-grid every 1 unit
ctx.strokeStyle='rgba(60,50,35,.08)';ctx.lineWidth=0.3;
for(let g=-30;g<=30;g+=1){
if(g%5===0)continue;
const[x1,y1]=g2s(g,-30);const[x2,y2]=g2s(g,30);
ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
const[x3,y3]=g2s(-30,g);const[x4,y4]=g2s(30,g);
ctx.beginPath();ctx.moveTo(x3,y3);ctx.lineTo(x4,y4);ctx.stroke()}}

// Major axes
ctx.strokeStyle='rgba(200,169,110,.3)';ctx.lineWidth=1.5;
const[ax0,ay0]=g2s(0,-30);const[ax1,ay1]=g2s(0,30);
ctx.beginPath();ctx.moveTo(ax0,ay0);ctx.lineTo(ax1,ay1);ctx.stroke();
const[bx0,by0]=g2s(-30,0);const[bx1,by1]=g2s(30,0);
ctx.beginPath();ctx.moveTo(bx0,by0);ctx.lineTo(bx1,by1);ctx.stroke();

// Axis labels at edges
ctx.font='bold '+fs(24)+' monospace';ctx.fillStyle='rgba(200,170,110,.5)';ctx.textAlign='center';
for(let g=-30;g<=30;g+=10){
const[lx,ly]=g2s(g,-31.5);ctx.fillText(g,lx,ly);
const[lx2,ly2]=g2s(-32,g);ctx.textAlign='right';ctx.fillText(g,lx2,ly2);ctx.textAlign='center'}

// Grid axis names
ctx.font='bold '+fs(28)+' sans-serif';ctx.fillStyle='#c8a96e';
const[xLx,xLy]=g2s(0,-33.5);ctx.fillText('X (grid)',xLx,xLy);
ctx.save();const[yLx,yLy]=g2s(-34.5,0);ctx.translate(yLx,yLy);ctx.rotate(-Math.PI/2);ctx.fillText('Y (grid)',0,0);ctx.restore();

// ═══ ANGULAR TICKS (0°-360° around scope boundary) ═══
ctx.font=fs(18)+' monospace';ctx.fillStyle='rgba(200,169,110,.4)';ctx.textAlign='center';
for(let d=0;d<360;d+=30){
const a=d*Math.PI/180;const tr=31.5;
const[tx,ty]=g2s(tr*Math.cos(a),tr*Math.sin(a));
ctx.fillText(d+'\u00b0',tx,ty);
// Tick mark
const[t1x,t1y]=g2s(29.5*Math.cos(a),29.5*Math.sin(a));
const[t2x,t2y]=g2s(30.5*Math.cos(a),30.5*Math.sin(a));
ctx.beginPath();ctx.strokeStyle='rgba(200,169,110,.3)';ctx.lineWidth=1;
ctx.moveTo(t1x,t1y);ctx.lineTo(t2x,t2y);ctx.stroke()}

// ═══ REFERENCE ORBIT RINGS (mapped to grid) ═══
const orbits=[[0.72,'rgba(210,180,100,.2)','Venus'],[1.0,'rgba(120,170,255,.25)','Earth'],[1.52,'rgba(255,120,90,.2)','Mars']];
for(const[rau,col,lbl]of orbits){
const gr=rau/Rs*30;
ctx.beginPath();for(let i=0;i<=64;i++){const a=i*Math.PI*2/64;
const[px,py]=g2s(gr*Math.cos(a),gr*Math.sin(a));
if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py)}
ctx.strokeStyle=col;ctx.lineWidth=1;ctx.stroke();
const[olx,oly]=g2s(gr+1.5,1);
ctx.font=fs(16)+' sans-serif';ctx.fillStyle=col;ctx.textAlign='left';ctx.fillText(lbl+' ('+rau+' AU)',olx,oly)}

// ═══ FTOP at grid origin ═══
const[ox,oy]=g2s(0,0);
ctx.save();ctx.shadowColor='#ffaa00';ctx.shadowBlur=15;
ctx.beginPath();ctx.arc(ox,oy,8,0,Math.PI*2);ctx.fillStyle='#ffcc00';ctx.fill();ctx.restore();
ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle='#ffcc00';ctx.textAlign='center';
ctx.fillText('\u25c7 FTOP (0,0)',ox,oy-14);
const ftopSph=au2sph(0,0);
ctx.font=fs(16)+' monospace';ctx.fillStyle='#aa8';ctx.fillText(ftopSph.addr,ox,oy+22);

// ═══ TRAJECTORY in grid coords ═══
ctx.beginPath();ctx.strokeStyle='#44ff88';ctx.lineWidth=2.5;ctx.shadowColor='#44ff88';ctx.shadowBlur=6;
for(let i=0;i<NT.length;i++){const gx=NT[i][0]/Rs*30,gy=NT[i][1]/Rs*30;
const[sx,sy]=g2s(gx,gy);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.shadowBlur=0;

// ═══ GATE DOTS with sphere addresses ═══
const FIB_SET=new Set([1,2,3,5,8]);
for(let k=0;k<GT.length;k++){
const gx=GT[k][0]/Rs*30,gy=GT[k][1]/Rs*30;
const[sx,sy]=g2s(gx,gy);const isFib=FIB_SET.has(k+1);
const gr=isFib?10:6;
if(isFib){ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=10;
ctx.beginPath();ctx.arc(sx,sy,gr+4,0,Math.PI*2);ctx.strokeStyle='rgba(255,170,60,.5)';ctx.lineWidth=2;ctx.stroke();ctx.restore()}
ctx.beginPath();ctx.arc(sx,sy,gr,0,Math.PI*2);ctx.fillStyle='#ffd700';ctx.fill();
ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke();
ctx.fillStyle='#1a1610';ctx.font='bold '+fs(isFib?20:16)+' sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(k+1,sx,sy);
// Sphere address label
const gsph=au2sph(GT[k][0],GT[k][1]);
ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(255,215,0,.5)';ctx.textBaseline='alphabetic';
ctx.fillText(gsph.addr,sx+gr+6,sy-4);
ctx.fillStyle='rgba(180,180,180,.4)';ctx.fillText('G'+(k+1)+' '+GI[k].spd.toFixed(1)+' AU/yr',sx+gr+6,sy+12)}

// ═══ BARREL ═══
const bsph=au2sph(B[0],B[1]);
const[bsx,bsy]=g2s(B[0]/Rs*30,B[1]/Rs*30);
ctx.fillStyle='#4488ff';ctx.fillRect(bsx-8,bsy-8,16,16);ctx.strokeStyle='#88bbff';ctx.lineWidth=2;ctx.strokeRect(bsx-8,bsy-8,16,16);
ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle='#66aaff';ctx.textAlign='center';
ctx.fillText('BARREL',bsx,bsy+28);
ctx.font=fs(16)+' monospace';ctx.fillStyle='#88bbff';ctx.fillText(bsph.addr,bsx,bsy+46);

// ═══ TARGET ═══
const tsph=au2sph(TG[0],TG[1]);
const[tsx,tsy]=g2s(TG[0]/Rs*30,TG[1]/Rs*30);
ctx.save();ctx.shadowColor='#44ff88';ctx.shadowBlur=18;
ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.8)';ctx.fill();ctx.restore();
ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(tsx,tsy,12,0,Math.PI*2);ctx.stroke();
ctx.font='bold '+fs(24)+' sans-serif';ctx.fillStyle='#44ff88';ctx.textAlign='center';
ctx.fillText('TARGET',tsx,tsy-20);
ctx.font=fs(16)+' monospace';ctx.fillStyle='#44ff88';ctx.fillText(tsph.addr,tsx,tsy+28);
// Target orbital history path in grid coords (orange — past trajectory)
if(TGT_ORB&&TGT_ORB.history){
ctx.beginPath();ctx.strokeStyle='rgba(255,140,60,.18)';ctx.lineWidth=1.5;ctx.setLineDash([3,5]);
const hStep=Math.max(1,Math.floor(TGT_ORB.history.length/200));
for(let hi=0;hi<TGT_ORB.history.length;hi+=hStep){const hgx=TGT_ORB.history[hi][0]/Rs*30,hgy=TGT_ORB.history[hi][1]/Rs*30;
const[hpx,hpy]=g2s(hgx,hgy);if(hi===0)ctx.moveTo(hpx,hpy);else ctx.lineTo(hpx,hpy)}
ctx.stroke();ctx.setLineDash([])}
// Target orbital forward path in grid coords (green — predicted)
if(TGT_ORB&&TGT_ORB.path){
ctx.beginPath();ctx.strokeStyle='rgba(68,255,136,.2)';ctx.lineWidth=1.5;ctx.setLineDash([5,5]);
const tStep=Math.max(1,Math.floor(TGT_ORB.path.length/200));
for(let ti=0;ti<TGT_ORB.path.length;ti+=tStep){const tgx2=TGT_ORB.path[ti][0]/Rs*30,tgy2=TGT_ORB.path[ti][1]/Rs*30;
const[tpx,tpy]=g2s(tgx2,tgy2);if(ti===0)ctx.moveTo(tpx,tpy);else ctx.lineTo(tpx,tpy)}
ctx.stroke();ctx.setLineDash([]);
// Target at T_f in grid
const tfgx=TGT_ORB.at_tf[0]/Rs*30,tfgy=TGT_ORB.at_tf[1]/Rs*30;
const[tfsx,tfsy]=g2s(tfgx,tfgy);
ctx.beginPath();ctx.arc(tfsx,tfsy,7,0,Math.PI*2);ctx.strokeStyle='#44ff88';ctx.lineWidth=2;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
const tfSph=au2sph(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(68,255,136,.5)';ctx.textAlign='center';
ctx.fillText('T@Tf '+tfSph.addr,tfsx,tfsy+18);
// Target gate-time dots
for(let tgi=0;tgi<TGT_ORB.gates.length;tgi++){const tggx=TGT_ORB.gates[tgi][0]/Rs*30,tggy=TGT_ORB.gates[tgi][1]/Rs*30;
const[tgsx2,tgsy2]=g2s(tggx,tggy);
ctx.beginPath();ctx.arc(tgsx2,tgsy2,3,0,Math.PI*2);ctx.fillStyle='rgba(68,255,136,.25)';ctx.fill()}}
// Energy label on barrel
if(ENER){ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(100,170,255,.5)';ctx.textAlign='center';
ctx.fillText('|v\u2080|='+ENER.v0_kms+' km/s | dist='+ENER.dist_au+' AU',bsx,bsy+62)}

// ═══ PARALLEL trajectory ═══
if(showParallel&&PAR){
ctx.beginPath();ctx.strokeStyle='rgba(0,220,120,.5)';ctx.lineWidth=1.5;ctx.setLineDash([5,4]);
for(let i=0;i<PAR.nt.length;i++){const gx=PAR.nt[i][0]/Rs*30,gy=PAR.nt[i][1]/Rs*30;
const[sx,sy]=g2s(gx,gy);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([])}

// ═══ SWARM trajectories ═══
if(showSwarm&&SWM){
for(let si=0;si<SWM.length;si++){const sw=SWM[si];const col=SWM_COLS[si%SWM_COLS.length];
ctx.beginPath();ctx.strokeStyle=col+'88';ctx.lineWidth=1;ctx.setLineDash([3,5]);
for(let i=0;i<sw.nt.length;i++){const gx=sw.nt[i][0]/Rs*30,gy=sw.nt[i][1]/Rs*30;
const[sx,sy]=g2s(gx,gy);if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy)}
ctx.stroke();ctx.setLineDash([]);
// Swarm barrel dot
const[sbx,sby]=g2s(sw.bp[0]/Rs*30,sw.bp[1]/Rs*30);
ctx.beginPath();ctx.arc(sbx,sby,5,0,Math.PI*2);ctx.fillStyle=col;ctx.fill();
ctx.font='bold '+fs(16)+' sans-serif';ctx.fillStyle=col;ctx.textAlign='center';
ctx.fillText('S@'+sw.ck+'h',sbx,sby+16)}}

// ═══ ANIMATED COMET ═══
if(animT>0){const ap=getAnimPos();const gx=ap[0]/Rs*30,gy=ap[1]/Rs*30;
const[asx,asy]=g2s(gx,gy);
ctx.save();ctx.shadowColor='#ff8800';ctx.shadowBlur=25;
ctx.beginPath();ctx.arc(asx,asy,9,0,Math.PI*2);ctx.fillStyle='#ffaa00';ctx.fill();ctx.restore();
ctx.beginPath();ctx.arc(asx,asy,9,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
const csph=au2sph(ap[0],ap[1]);
ctx.font='bold '+fs(24)+' monospace';ctx.fillStyle='#ffaa00';ctx.textAlign='left';
ctx.fillText('T+'+animT.toFixed(3)+' yr',asx+16,asy);
ctx.font=fs(18)+' monospace';ctx.fillStyle='#ffd700';ctx.fillText(csph.addr,asx+16,asy+22);
ctx.fillStyle='#889';ctx.fillText('Cycle: '+csph.time,asx+16,asy+40)}

// ═══ QUADRANT LABELS ═══
ctx.font=fs(14)+' monospace';ctx.fillStyle='rgba(100,100,130,.4)';ctx.textAlign='center';
const qlOff=22;
const[q1x,q1y]=g2s(-qlOff,qlOff);ctx.fillText('-30z,30x,-30y',q1x,q1y);
const[q2x,q2y]=g2s(qlOff,qlOff);ctx.fillText('-30z,30x,30y',q2x,q2y);
const[q3x,q3y]=g2s(-qlOff,-qlOff);ctx.fillText('30z,30x,-30y',q3x,q3y);
const[q4x,q4y]=g2s(qlOff,-qlOff);ctx.fillText('30z,-30x,30y',q4x,q4y);

// ═══ TITLE ═══
ctx.font='bold '+fs(34)+' "Times New Roman",serif';ctx.fillStyle='#c8a96e';ctx.textAlign='center';
ctx.fillText('Dimensional Sphere Navigation \u2014 X\u00b0,xz,xx,xy',w/2,34);
ctx.font=fs(22)+' sans-serif';ctx.fillStyle='#889';
ctx.fillText('5,184,000 sphere locations | \u221e = Scope Rs ('+Rs+' AU) | Grid \u00b130 | 24 cycles | Scroll=zoom | Drag=pan',w/2,60);

// ═══ INFO PANEL (right side) ═══
let si='<div style="font:bold 12px sans-serif;color:#ffd700;margin-bottom:6px;border-bottom:1px solid #5a4a32;padding-bottom:4px">\u25c9 SPHERE COORDINATE MAP</div>';
si+='<div style="font-size:9px;color:#889;margin-bottom:6px">';
si+='<b style="color:#c8a96e">Format:</b> X\u00b0,xz,xx,xy = location<br>';
si+='<b style="color:#c8a96e">Dim:</b> 0\u00b0-360\u00b0, 0:00:00-23:59:59<br>';
si+='<b style="color:#c8a96e">Grid:</b> ((60\u00d760\u00d760)\u00d724) = 5,184,000<br>';
si+='<b style="color:#c8a96e">\u221e:</b> Rs='+Rs+' AU \u2192 \u00b130 grid units<br>';
si+='<b style="color:#c8a96e">Scale:</b> 1 grid unit = '+(Rs/30).toFixed(4)+' AU</div>';
si+='<div style="border-top:1px solid #3a3020;padding-top:4px;font-size:9px">';
si+='<b style="color:#ffd700">BARREL</b><br>'+bsph.addr+' | Cycle: '+bsph.time+'<br>';
si+='<b style="color:#44ff88">TARGET</b><br>'+tsph.addr+' | Cycle: '+tsph.time+'<br>';
if(animT>0){const csph2=au2sph(getAnimPos()[0],getAnimPos()[1]);
si+='<b style="color:#ffaa00">COMET</b><br>'+csph2.addr+' | '+csph2.time+'<br>'}
si+='</div>';
si+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px;color:#889">';
si+='<b style="color:#c8a96e">GATES (sphere addresses):</b><br>';
for(let k=0;k<GT.length;k++){const gs=au2sph(GT[k][0],GT[k][1]);
si+='<span style="color:'+(FIB_SET.has(k+1)?'#ffd700':'#aa9')+'">G'+(k+1)+': '+gs.addr+'</span><br>'}
si+='</div>';
// Target orbital motion section
if(TGT_ORB){
si+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px">';
si+='<b style="color:#44ff88">\u2609 TARGET MOTION</b><br>';
si+='<span style="color:#889">v_orb:</span> <span style="color:#44ff88">'+TGT_ORB.spd+' AU/yr ('+TGT_ORB.spd_kms+' km/s)</span><br>';
si+='<span style="color:#889">Period:</span> <span style="color:#44ff88">'+TGT_ORB.period+' yr</span><br>';
si+='<span style="color:#889">At T_f:</span> <span style="color:#44ff88">('+TGT_ORB.at_tf[0].toFixed(3)+', '+TGT_ORB.at_tf[1].toFixed(3)+')</span><br>';
const tSph=au2sph(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
si+='<span style="color:#889">At T_f (sph):</span> <span style="color:#ffd700">'+tSph.addr+'</span><br>';
si+='</div>'}
// Energy / mass model
if(ENER){
si+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px">';
si+='<b style="color:#66aaff">\u26a1 ENERGY MODEL</b><br>';
si+='<span style="color:#889">Launch |v\u2080|:</span> <span style="color:#ffd700">'+ENER.v0_mag+' AU/yr ('+ENER.v0_kms+' km/s)</span><br>';
si+='<span style="color:#889">B\u2192T dist:</span> <span style="color:#ffd700">'+ENER.dist_au+' AU</span><br>';
for(const[lbl,ed] of Object.entries(ENER.energy)){
si+='<span style="color:#aa9">'+lbl+':</span> <span style="color:#ff8800">'+ed.ke_j.toExponential(1)+' J ('+ed.ke_tnt+' kT)</span><br>'}
si+='</div>';
si+='<div style="border-top:1px solid #3a3020;padding-top:4px;margin-top:4px;font-size:9px">';
si+='<b style="color:#cc44ff">\u2604 ENVIRONMENT</b><br>';
si+='<span style="color:#889">Solar v(galactic):</span> '+ENER.galactic.solar_v_galactic_kms+' km/s<br>';
si+='<span style="color:#889">Solar v(LSR):</span> '+ENER.galactic.solar_v_lsr_kms+' km/s<br>';
si+='<span style="color:#889">ISM density:</span> '+ENER.galactic.local_ism_density_cm3+' /cm\u00b3<br>';
si+='<span style="color:#889">Rad. pressure(1AU):</span> '+ENER.galactic.radiation_pressure_au1_Nm2+' N/m\u00b2<br>';
si+='</div>'}
document.getElementById('sphInfo').innerHTML=si;

// ═══ MOUSE COORDINATE READOUT ═══
if(c._sphMouse){const[mgx,mgy]=c._sphMouse;
const mau_x=mgx/30*Rs,mau_y=mgy/30*Rs;
const msph=au2sph(mau_x,mau_y);
const mr=Math.sqrt(mgx*mgx+mgy*mgy);
const inScope=mr<=30;
document.getElementById('sphCoord').innerHTML=
'<span style="color:'+(inScope?'#ffd700':'#ff4444')+'">'+msph.addr+'</span> | '+
'AU: ('+mau_x.toFixed(4)+', '+mau_y.toFixed(4)+') | '+
'Grid: ('+mgx.toFixed(1)+', '+mgy.toFixed(1)+') | '+
'R: '+mr.toFixed(1)+'/30 | '+
'Cycle: '+msph.time+
(inScope?'':' <span style="color:#ff4444">[OUTSIDE \u221e]</span>')}
}

// Sphere nav mouse handlers
(function(){
const sc2=document.getElementById('sphC');
sc2.addEventListener('mousemove',e=>{
const rc=sc2.getBoundingClientRect();const mx=e.clientX-rc.left,my=e.clientY-rc.top;
const w=sc2.width,h=sc2.height,cx=w/2,cy=h/2;
const scopeR=Math.min(w,h)/2.3*sphZoom;
const gx=(mx-cx)/scopeR*30+sphPanX*30;const gy=(cy-my)/scopeR*30+sphPanY*30;
sc2._sphMouse=[gx,gy];
if(sphDrg){sphPanX+=(e.clientX-sphLmx)/scopeR;sphPanY-=(e.clientY-sphLmy)/scopeR;sphLmx=e.clientX;sphLmy=e.clientY;draw()}
else{draw()}});
sc2.addEventListener('mousedown',e=>{if(e.button===2||e.button===0){sphDrg=true;sphLmx=e.clientX;sphLmy=e.clientY}});
sc2.addEventListener('mouseup',()=>sphDrg=false);
sc2.addEventListener('mouseleave',()=>{sphDrg=false});
sc2.addEventListener('wheel',e=>{
const f=e.deltaY>0?0.9:1.1;sphZoom=Math.max(0.3,Math.min(10,sphZoom*f));draw();e.preventDefault()},{passive:false});
sc2.addEventListener('contextmenu',e=>e.preventDefault());
})();

// CONFIG tab — fully customizable parameter panel
(function(){const cp=document.getElementById('cfgContent');
const v0mag=Math.sqrt(V0[0]*V0[0]+V0[1]*V0[1]);
const barAng=Math.atan2(B[1],B[0])*180/Math.PI;
const barR=Math.sqrt(B[0]*B[0]+B[1]*B[1]);
const tgtR=Math.sqrt(TG[0]*TG[0]+TG[1]*TG[1]);
const tgtAng=Math.atan2(TG[1],TG[0])*180/Math.PI;
let h='';
h+='<div class="cd"><h2>\u2699 MISSION CONFIGURATION</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:12px">Full parameter set for comet/meteor redirection. Edit values and click APPLY to re-simulate. All positions in AU, velocities in AU/yr, angles in degrees.</p>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">';
// BARREL (PUSHER) SECTION
h+='<div class="cd" style="border:1px solid #5a4a32"><h2 style="font-size:13px;color:#4488ff">\u25a0 PRIMARY BARREL / PUSHER</h2>';
h+='<p style="font:9px sans-serif;color:#88aacc;margin-bottom:8px">Place the barrel <b>anywhere</b> in AU space. The 7 o\'clock position (240\u00b0, r=Rs) is the default orientation focus &mdash; FTOP is always the manifest center. The barrel can be placed at any radius and angle; the Newton solver recomputes the full transfer orbit automatically.</p>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
h+='<div class="cd" style="border:1px solid #4488ff44"><div style="font:bold 9px sans-serif;color:#4488ff;margin-bottom:4px">POSITION (free placement)</div>';
h+='<div class="rw"><span class="lb">X (AU)</span><input class="cfg-in" id="cfg-bx" type="number" step="0.01" value="'+B[0].toFixed(4)+'" oninput="brlUpdate()"></div>';
h+='<div class="rw"><span class="lb">Y (AU)</span><input class="cfg-in" id="cfg-by" type="number" step="0.01" value="'+B[1].toFixed(4)+'" oninput="brlUpdate()"></div>';
h+='<div class="rw"><span class="lb">R from FTOP</span><span class="vl" id="brl-r">'+barR.toFixed(4)+' AU</span></div>';
h+='<div class="rw"><span class="lb">Angle</span><span class="vl" id="brl-ang">'+barAng.toFixed(1)+'\u00b0</span></div>';
h+='<div class="rw"><span class="lb">Clock equiv.</span><span class="vl" id="brl-clk">'+(Math.round(((90-barAng+360)%360)/30)||12)+'h</span></div>';
h+='</div>';
h+='<div class="cd" style="border:1px solid #4488ff44"><div style="font:bold 9px sans-serif;color:#4488ff;margin-bottom:4px">SNAP TO CLOCK POSITION</div>';
h+='<p style="font:8px sans-serif;color:#778;margin-bottom:4px">Snap barrel to exact scope ring clock position (r=Rs):</p>';
h+='<div class="rw"><span class="lb">Clock (1-12h)</span><input class="cfg-in" id="cfg-bclk" type="number" min="1" max="12" step="1" value="7"></div>';
h+='<button onclick="snapBarrel()" style="margin-top:6px;padding:4px 10px;background:#1a3060;border:1px solid #4488ff;color:#88aaff;border-radius:3px;cursor:pointer;font:10px sans-serif">SNAP TO CLOCK</button>';
h+='<div style="font:8px sans-serif;color:#556;margin-top:4px">Scope ring r='+Rs.toFixed(2)+' AU. 7h = 240\u00b0 (default focus orientation).</div>';
h+='</div></div>';
h+='<p style="font:9px sans-serif;color:#556;margin-top:8px">\u2139 Orientation: <b>FTOP</b> is coordinate origin (0,0). <b>7 o\'clock</b> (240\u00b0 from 12 o\'clock CW) is the canonical barrel focus position. When any barrel is focused in SALVO view, it renders at 7h and all others orbit relative to it. The barrel can be at <i>any</i> AU position &mdash; the solver finds the exact transfer orbit regardless of placement.</p>';
// === PROJECTILE / ENERGY (per-barrel default) ===
// Every barrel carries a default projectile spec so the power needed to make
// the hit is part of its knowledge. Diameter + density + Isp; mass is a sphere
// of uniform density; KE / total energy / average power follow from |v0|.
if(ENER&&ENER.primary_proj){const pr=ENER.primary_proj;const pd=ENER.proj_default||{diameter_m:100,density_kg_m3:2600,Isp_s:350};
h+='<div class="cd" style="border:1px solid #ffd70044;margin-top:10px"><div style="font:bold 10px sans-serif;color:#ffd700;margin-bottom:6px">\u25c6 PROJECTILE \u2014 default barrel knowledge</div>';
h+='<p style="font:9px sans-serif;color:#bbaa77;margin-bottom:6px">Diameter + density + specific-impulse drive the mass and energy/power budget. Apply re-runs the solver with the new spec.</p>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
h+='<div>';
h+='<div class="rw"><span class="lb">Diameter (m)</span><input class="cfg-in" id="cfg-pd" type="number" step="1" value="'+pd.diameter_m+'"></div>';
h+='<div class="rw"><span class="lb">Density (kg/m\u00b3)</span><input class="cfg-in" id="cfg-prho" type="number" step="50" value="'+pd.density_kg_m3+'"></div>';
h+='<div class="rw"><span class="lb">Isp (s)</span><input class="cfg-in" id="cfg-pisp" type="number" step="10" value="'+pd.Isp_s+'"></div>';
h+='<div class="rw"><span class="lb">Mass (computed)</span><span class="vl">'+pr.mass_kg.toExponential(3)+' kg</span></div>';
h+='</div><div>';
h+='<div class="rw"><span class="lb">KE at launch</span><span class="vl">'+pr.ke_j.toExponential(3)+' J</span></div>';
h+='<div class="rw"><span class="lb">TNT equiv.</span><span class="vl">'+pr.tnt_kt.toFixed(3)+' kt</span></div>';
h+='<div class="rw"><span class="lb">Total E (KE+\u0394GPE)</span><span class="vl">'+pr.total_j.toExponential(3)+' J</span></div>';
h+='<div class="rw"><span class="lb">Avg power over T<sub>f</sub></span><span class="vl" style="color:#ffaa44">'+pr.power_avg_W.toExponential(3)+' W</span></div>';
h+='<div class="rw"><span class="lb">\u0394v launch</span><span class="vl">'+pr.dv_launch_kms.toFixed(3)+' km/s</span></div>';
h+='<div class="rw"><span class="lb">\u0394v total</span><span class="vl">'+pr.dv_total_kms.toFixed(3)+' km/s</span></div>';
h+='<div class="rw"><span class="lb">Fuel fraction</span><span class="vl">'+(pr.fuel_frac*100).toFixed(3)+'%</span></div>';
h+='</div></div>';
h+='<p style="font:9px sans-serif;color:#667;margin-top:6px">Every SALVO barrel inherits this spec by default. The parallel and swarm barrels have their own live energy numbers in the SALVO info panel.</p>';
h+='</div>'}
// === PER-BARREL PROJECTILE LIST — one row per barrel, editable overrides ===
// Lists every projectile in the system (primary + parallel + swarm) so the user
// can see + edit which comet/asteroid-class body each barrel is throwing.
// Overrides are posted as `swarm_proj` on APPLY and kept with the barrel.
h+='<div class="cd" style="border:1px solid #ffaa4444;margin-top:10px"><h2 style="font-size:13px;color:#ffaa44">★ PROJECTILE LIST (comet → meteor)</h2>';
h+='<p style="font:9px sans-serif;color:#bbaa77;margin-bottom:8px">Every barrel ships with a projectile. Use this table to override diameter, density, or I<sub>sp</sub> per barrel. Mass, KE, TNT-equivalent and average power are recomputed automatically.</p>';
h+='<div style="overflow-x:auto"><table style="width:100%;font-size:10px;border-collapse:collapse">';
h+='<thead><tr style="color:#c8a96e;border-bottom:1px solid #5a4a32"><th style="text-align:left;padding:4px">Barrel</th><th>Preset</th><th>⌀ (m)</th><th>ρ (kg/m³)</th><th>Isp (s)</th><th>Mass (kg)</th><th>KE (J)</th><th>TNT (kt)</th><th>P<sub>avg</sub> (W)</th><th>Δv<sub>tot</sub> (km/s)</th><th>Fuel %</th></tr></thead><tbody>';
const PROJ_PRESETS=[['100m bolide',100,2600,350],['1km asteroid',1000,2600,350],['10km comet',10000,500,350],['1m probe',1,1000,450],['100kg inceptor',0.5,1000,300]];
// Helper to emit one row
function projRow(prefix,lbl,col,pr){
let row='<tr style="border-bottom:1px solid rgba(90,74,50,.12)">';
row+='<td style="color:'+col+';padding:4px"><b>'+lbl+'</b></td>';
row+='<td><select class="cfg-in" id="'+prefix+'_preset" style="width:118px">';
row+='<option value="">custom</option>';
for(const p of PROJ_PRESETS)row+='<option value="'+p[1]+','+p[2]+','+p[3]+'">'+p[0]+'</option>';
row+='</select></td>';
if(pr){
row+='<td><input class="cfg-in" id="'+prefix+'_d" type="number" step="1" value="'+pr.diameter_m+'" style="width:70px"></td>';
row+='<td><input class="cfg-in" id="'+prefix+'_rho" type="number" step="50" value="'+pr.density_kg_m3+'" style="width:74px"></td>';
row+='<td><input class="cfg-in" id="'+prefix+'_isp" type="number" step="10" value="'+pr.Isp_s+'" style="width:60px"></td>';
row+='<td>'+pr.mass_kg.toExponential(2)+'</td>';
row+='<td>'+pr.ke_j.toExponential(2)+'</td>';
row+='<td>'+pr.tnt_kt.toFixed(2)+'</td>';
row+='<td style="color:#ffaa44">'+pr.power_avg_W.toExponential(2)+'</td>';
row+='<td>'+pr.dv_total_kms.toFixed(2)+'</td>';
row+='<td>'+(pr.fuel_frac*100).toFixed(2)+'</td>'}
else{row+='<td colspan="9" style="color:#667;padding:4px">no projectile data</td>'}
row+='</tr>';return row}
if(ENER&&ENER.primary_proj)h+=projRow('proj0','PRIMARY','#ffd700',ENER.primary_proj);
if(PAR&&PAR.proj)h+=projRow('projP','PARALLEL +'+PAR.off+'°','#00dd88',PAR.proj);
for(let si=0;si<SWM.length;si++){h+=projRow('projS'+si,'SWARM @'+SWM[si].ck+'h',SWM_COLS[si%SWM_COLS.length],SWM[si].proj||null)}
h+='</tbody></table></div>';
h+='<div style="font:9px sans-serif;color:#667;margin-top:8px;padding:6px;border:1px dashed #5a4a32;border-radius:4px"><b style="color:#c8a96e">Turning comet into meteor:</b> a comet is a natural body on its own Keplerian orbit; a meteor is that body after the barrel has nudged it with Δv so it impacts the target. The table above sets the body each barrel is nudging. Apply re-runs the full solver so every Δv budget, power need, and fuel fraction updates.</div>';
h+='<div style="font:9px sans-serif;color:#667;margin-top:6px">Preset dropdown fills the three numeric fields; set the preset to &ldquo;custom&rdquo; and edit manually for non-standard bodies. Leaving a cell blank inherits the default projectile.</div>';
h+='</div>';
h+='</div>';
// TARGET SECTION
h+='<div class="cd" style="border:1px solid #44ff8866"><h2 style="font-size:13px;color:#44ff88">\u25cf TARGET</h2>';
h+='<p style="font:9px sans-serif;color:#88ccaa;margin-bottom:8px">Target position in world AU coordinates. The Newton solver computes the exact intercept orbit. All derived fields auto-update.</p>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">';
h+='<div class="cd" style="border:1px solid #44ff8844"><div style="font:bold 9px sans-serif;color:#44ff88;margin-bottom:4px">POSITION (free placement)</div>';
h+='<div class="rw"><span class="lb">X (AU)</span><input class="cfg-in" id="cfg-tx" type="number" step="0.01" value="'+TG[0].toFixed(4)+'" oninput="tgtUpdate()"></div>';
h+='<div class="rw"><span class="lb">Y (AU)</span><input class="cfg-in" id="cfg-ty" type="number" step="0.01" value="'+TG[1].toFixed(4)+'" oninput="tgtUpdate()"></div>';
h+='<div class="rw"><span class="lb">Z (AU, 3D only)</span><input class="cfg-in" id="cfg-tz" type="number" step="0.01" value="'+(TGT_Z||0).toFixed(4)+'" oninput="tgtUpdate()"></div>';
h+='<div class="rw" title="Include all 20 planets + moons as gravitating sources. Slow — expect a ~30 s wait when enabled."><span class="lb">N-body perturbations</span><input class="cfg-in" id="cfg-pert" type="checkbox" style="width:auto;margin:3px 4px" '+(USE_PERT?'checked':'')+'><span style="font:9px sans-serif;color:#667">(slow; off by default)</span></div>';
h+='<div class="rw"><span class="lb">R(FTOP)</span><span class="vl" id="tgt-r">'+tgtR.toFixed(4)+' AU</span></div>';
h+='<div class="rw"><span class="lb">Angle</span><span class="vl" id="tgt-ang">'+tgtAng.toFixed(1)+'\u00b0</span></div>';
h+='<div class="rw"><span class="lb">Clock equiv.</span><span class="vl" id="tgt-clk">'+(Math.round(((90-tgtAng+360)%360)/30)||12)+'h</span></div>';
h+='<div class="rw"><span class="lb">Grid coord</span><span class="vl" id="tgt-grid">('+( TG[0]/Rs*30).toFixed(1)+', '+(TG[1]/Rs*30).toFixed(1)+')</span></div>';
h+='</div>';
h+='<div class="cd" style="border:1px solid #44ff8844"><div style="font:bold 9px sans-serif;color:#44ff88;margin-bottom:4px">HIT GEOMETRY</div>';
h+='<div class="rw"><span class="lb">Hit radius (AU)</span><input class="cfg-in" id="cfg-hr" type="number" step="0.001" value="'+HR.toFixed(4)+'"></div>';
h+='<div class="rw"><span class="lb">Hit radius (km)</span><span class="vl">'+(HR*1.496e8).toFixed(0)+' km</span></div>';
h+='<div class="rw"><span class="lb">Hit radius (grid)</span><span class="vl">'+(HR/Rs*30).toFixed(4)+' units</span></div>';
const gapAU=Math.sqrt((TG[0]-B[0])**2+(TG[1]-B[1])**2);
const gapKm=gapAU*1.496e8;
h+='<div class="rw"><span class="lb">B\u2192T distance</span><span class="vl" style="color:#ffd700">'+gapAU.toFixed(4)+' AU ('+gapKm.toExponential(2)+' km)</span></div>';
h+='<div class="rw"><span class="lb">Hit angular size</span><span class="vl">'+(2*Math.atan(HR/gapAU)*180/Math.PI*3600).toFixed(1)+' arcsec</span></div>';
h+='<div class="rw"><span class="lb">Status</span><span class="vl" style="color:#44ff88">\u2714 Active target</span></div>';
h+='</div></div>';
h+='<div style="border-top:1px solid #3a3020;margin-top:8px;padding-top:6px">';
h+='<div style="font:bold 10px sans-serif;color:#88ccaa;margin-bottom:4px">TARGET ORBITAL MOTION</div>';
h+='<div class="rw"><span class="lb">v\u2093 (AU/yr)</span><span class="vl">'+TGT_ORB.v0[0].toFixed(4)+'</span></div>';
h+='<div class="rw"><span class="lb">v\u1d67 (AU/yr)</span><span class="vl">'+TGT_ORB.v0[1].toFixed(4)+'</span></div>';
h+='<div class="rw"><span class="lb">Orbital speed</span><span class="vl">'+TGT_ORB.spd+' AU/yr ('+TGT_ORB.spd_kms+' km/s)</span></div>';
h+='<div class="rw"><span class="lb">Period</span><span class="vl">'+TGT_ORB.period+' yr</span></div>';
h+='<div class="rw"><span class="lb">At T\u2086 (intercept)</span><span class="vl">('+TGT_ORB.at_tf[0].toFixed(4)+', '+TGT_ORB.at_tf[1].toFixed(4)+') AU</span></div>';
h+='<div class="rw"><span class="lb">\u03b2 (v/c)</span><span class="vl">'+TGT_ORB.beta.toExponential(4)+'</span></div>';
h+='<div class="rw"><span class="lb">\u03b3 (Lorentz)</span><span class="vl">'+TGT_ORB.gamma.toFixed(10)+'</span></div>';
const tsfSph=au2sph(TGT_ORB.at_tf[0],TGT_ORB.at_tf[1]);
h+='<div class="rw"><span class="lb">At T\u2086 sphere</span><span class="vl" style="color:#ffd700">'+tsfSph.addr+'</span></div>';
h+='</div>';
h+='<p style="font:9px sans-serif;color:#556;margin-top:6px">Target is in world AU coordinates. It pans and zooms with the scope (world-coord camera). Click APPLY to re-simulate with new target.</p>';
h+='</div></div>';
h+='</div>';
// COMET / METEOR OBJECT
h+='<div class="cd"><h2>\u2604 COMET / METEOR OBJECT</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">Define the projectile properties. Load a simple comet, then adjust pathing to convert it into a meteor redirect.</p>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">';
h+='<div class="cd" style="border:1px solid #ffaa4466"><h3 style="font-size:11px;color:#ffaa44">INITIAL STATE</h3>';
h+='<div class="rw"><span class="lb">v\u2093 (AU/yr)</span><input class="cfg-in" id="cfg-vx" type="number" step="0.01" value="'+V0[0].toFixed(4)+'"></div>';
h+='<div class="rw"><span class="lb">v\u1d67 (AU/yr)</span><input class="cfg-in" id="cfg-vy" type="number" step="0.01" value="'+V0[1].toFixed(4)+'"></div>';
h+='<div class="rw"><span class="lb">|v\u2080| (AU/yr)</span><span class="vl" style="color:#ffd700">'+v0mag.toFixed(4)+'</span></div>';
h+='<div class="rw"><span class="lb">|v\u2080| (km/s)</span><span class="vl" style="color:#ffd700">'+(v0mag*4.74047).toFixed(2)+'</span></div>';
const v0hdg=(Math.atan2(V0[1],V0[0])*180/Math.PI);
h+='<div class="rw"><span class="lb">Heading</span><span class="vl">'+v0hdg.toFixed(1)+'\u00b0</span></div>';
h+='<div class="rw"><span class="lb">\u03b2 (v\u2080/c)</span><span class="vl">'+(v0mag/63241.1).toExponential(4)+'</span></div>';
h+='<div class="rw"><span class="lb">\u03b3</span><span class="vl">'+(1/Math.sqrt(1-(v0mag/63241.1)**2)).toFixed(10)+'</span></div>';
h+='</div>';
h+='<div class="cd" style="border:1px solid #ff886666"><h3 style="font-size:11px;color:#ff8866">PHYSICAL PROPERTIES</h3>';
h+='<div class="rw"><span class="lb">Mass (kg)</span><input class="cfg-in" id="cfg-mass" type="number" step="1e10" value="1e13"></div>';
h+='<div class="rw"><span class="lb">Diameter (m)</span><input class="cfg-in" id="cfg-diam" type="number" step="100" value="5000"></div>';
h+='<div class="rw"><span class="lb">Albedo</span><input class="cfg-in" id="cfg-alb" type="number" step="0.01" value="0.04"></div>';
h+='<div class="rw"><span class="lb">Type</span><select class="cfg-in" id="cfg-type"><option value="comet" selected>Comet (icy)</option><option value="asteroid">Asteroid (rocky)</option><option value="metallic">Metallic</option><option value="custom">Custom</option></select></div>';
h+='<div class="rw"><span class="lb">Density (kg/m\u00b3)</span><span class="vl">'+(1e13/(4/3*Math.PI*Math.pow(5000/2,3))).toFixed(1)+'</span></div>';
h+='<div class="rw"><span class="lb">Escape vel (m/s)</span><span class="vl">'+(Math.sqrt(2*6.674e-11*1e13/(5000/2))).toFixed(2)+'</span></div>';
h+='<div class="rw"><span class="lb">KE at |v\u2080| (J)</span><span class="vl" style="color:#ff8866">'+(0.5*1e13*Math.pow(v0mag*4740.47,2)).toExponential(2)+'</span></div>';
h+='</div>';
h+='<div class="cd" style="border:1px solid #4488ff66"><h3 style="font-size:11px;color:#4488ff">PATHING MODE</h3>';
h+='<div class="rw"><span class="lb">Redirect mode</span><select class="cfg-in" id="cfg-mode"><option value="newton">Newton shooting</option><option value="lambert">Lambert transfer</option><option value="manual">Manual v\u2080</option></select></div>';
h+='<div class="rw"><span class="lb">Correction type</span><select class="cfg-in" id="cfg-corr"><option value="jacobian" selected>Jacobian \u0394v</option><option value="none">None (ballistic)</option><option value="optimal">Optimal control</option></select></div>';
h+='<div class="rw"><span class="lb">Object role</span><select class="cfg-in" id="cfg-role"><option value="comet">Comet (natural)</option><option value="meteor" selected>Meteor (redirected)</option><option value="probe">Probe (artificial)</option></select></div>';
h+='<p style="font:9px sans-serif;color:#556;margin-top:6px">Comet \u2192 Meteor: Load comet orbit, apply \u0394v at barrel to redirect toward target. The correction gates refine the path.</p>';
h+='</div></div></div>';
// FLIGHT & PHYSICS
h+='<div class="cd"><h2>\u2697 FLIGHT & PHYSICS PARAMETERS</h2>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">';
h+='<div>';
h+='<div class="rw"><span class="lb">Flight time (yr)</span><input class="cfg-in" id="cfg-tf" type="number" step="0.01" value="'+TF+'"></div>';
h+='<div class="rw"><span class="lb">Integration dt (yr)</span><input class="cfg-in" id="cfg-dt" type="number" step="0.001" value="0.005"></div>';
h+='<div class="rw"><span class="lb">dt (days)</span><span class="vl">'+(0.005*365.25).toFixed(2)+'</span></div>';
h+='<div class="rw"><span class="lb">\u03bc = GM (AU\u00b3/yr\u00b2)</span><input class="cfg-in" id="cfg-mu" type="number" step="0.1" value="'+(4*Math.PI*Math.PI).toFixed(6)+'"></div>';
h+='<div class="rw"><span class="lb">\u03bc in SI (m\u00b3/s\u00b2)</span><span class="vl">1.327\u00d710\u00b2\u2070</span></div>';
h+='</div><div>';
h+='<div class="rw"><span class="lb">Number of gates</span><input class="cfg-in" id="cfg-ng" type="number" min="1" max="36" value="12"></div>';
h+='<div class="rw"><span class="lb">Gate spacing (\u00b0)</span><span class="vl">30\u00b0 (360/12)</span></div>';
h+='<div class="rw"><span class="lb">Gate \u0394t (yr)</span><span class="vl" style="color:#ffd700">'+(TF/13).toFixed(5)+' yr ('+(TF/13*365.25).toFixed(2)+' d)</span></div>';
h+='<div class="rw"><span class="lb">Damping factor</span><input class="cfg-in" id="cfg-damp" type="number" step="0.05" value="0.7"></div>';
h+='<div class="rw"><span class="lb">Max \u0394v/gate (AU/yr)</span><input class="cfg-in" id="cfg-maxdv" type="number" step="0.005" value="0.05"></div>';
h+='<div class="rw"><span class="lb">Scope radius Rs (AU)</span><input class="cfg-in" id="cfg-rs" type="number" step="0.1" value="'+Rs.toFixed(2)+'"></div>';
h+='<div class="rw"><span class="lb">Rs (km)</span><span class="vl">'+(Rs*1.496e8).toExponential(3)+' km</span></div>';
h+='<div class="rw"><span class="lb">Grid scale</span><span class="vl">1 unit = '+(Rs/30).toFixed(4)+' AU</span></div>';
h+='</div></div>';
// GATE JACOBIAN TABLE
h+='<div style="margin-top:12px;overflow-x:auto"><div style="font:bold 10px sans-serif;color:#ffd700;margin-bottom:6px">GATE JACOBIAN SUMMARY TABLE</div>';
h+='<table style="width:100%;font-size:9px;border-collapse:collapse;text-align:right">';
h+='<tr style="color:#ffd700;border-bottom:1px solid #5a4a32"><th style="text-align:left">G</th><th>Position</th><th>Speed</th><th>R(FTOP)</th><th>Heading</th><th>J-cond</th><th>Curv</th><th>\u0394hdg</th><th>\u0394spd</th></tr>';
for(const g of GI){const isFib=SAL_FIB.has(g.n);
h+='<tr style="border-bottom:1px solid rgba(90,74,50,.2);color:'+(isFib?'#ffd700':'#aa9')+'">';
h+='<td style="text-align:left;font-weight:bold">'+(isFib?'F':'')+g.n+'</td>';
h+='<td>('+g.pos[0]+', '+g.pos[1]+')</td><td>'+g.spd+'</td><td>'+g.rsun+'</td><td>'+g.hdg+'\u00b0</td><td>'+g.jc+'</td><td>'+g.curv+'</td><td>'+g.dhdg+'\u00b0</td><td>'+(g.dspd>=0?"+":"")+(g.dspd).toFixed(2)+'</td></tr>'}
h+='</table></div></div>';
// MONTE CARLO & NOISE
h+='<div class="cd"><h2>\u2684 MONTE CARLO & NOISE MODEL</h2>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">';
h+='<div>';
h+='<div class="rw"><span class="lb">Simulations</span><input class="cfg-in" id="cfg-ns" type="number" step="1000" value="'+R.n+'"></div>';
h+='<div class="rw"><span class="lb">Position noise \u03c3 (AU)</span><input class="cfg-in" id="cfg-sigp" type="number" step="0.001" value="0.01"></div>';
h+='<div class="rw"><span class="lb">Velocity noise \u03c3 (AU/yr)</span><input class="cfg-in" id="cfg-sigv" type="number" step="0.0001" value="0.001"></div>';
h+='</div><div>';
h+='<div class="rw"><span class="lb">Noise distribution</span><select class="cfg-in" id="cfg-ndist"><option value="gaussian" selected>Gaussian</option><option value="uniform">Uniform</option><option value="cauchy">Cauchy (heavy tail)</option></select></div>';
h+='<div class="rw"><span class="lb">Seed</span><input class="cfg-in" id="cfg-seed" type="number" value="42"></div>';
h+='<div class="rw"><span class="lb">Baseline campaign</span><select class="cfg-in" id="cfg-bcamp"><option value="yes" selected>Yes</option><option value="no">No (skip)</option></select></div>';
h+='</div></div></div>';
// PARALLEL SCOPE
h+='<div class="cd"><h2 style="color:#00dd88">\u2225 PARALLEL OFFSET SCOPE</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">A second scope offset by a configurable angle. The difference between primary and parallel trajectories at each gate provides live corrective sensitivity data.</p>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">';
h+='<div>';
h+='<div class="rw"><span class="lb">Offset angle (\u00b0)</span><input class="cfg-in" id="cfg-poff" type="number" step="0.5" value="'+PAR.off+'"></div>';
h+='<div class="rw"><span class="lb">Parallel barrel</span><span class="vl">('+PAR.bp[0].toFixed(4)+', '+PAR.bp[1].toFixed(4)+') AU</span></div>';
h+='<div class="rw"><span class="lb">Parallel v\u2080</span><span class="vl">('+PAR.v0[0].toFixed(4)+', '+PAR.v0[1].toFixed(4)+') AU/yr</span></div>';
h+='<div class="rw"><span class="lb">Parallel miss</span><span class="vl" style="color:#00dd88">'+PAR.miss+' AU</span></div>';
h+='</div><div>';
h+='<div style="font:bold 10px sans-serif;color:#889;margin-bottom:6px">GATE \u0394 TABLE (Par - Primary)</div>';
h+='<div style="overflow-x:auto"><table style="width:100%;font-size:9px;border-collapse:collapse">';
h+='<tr style="color:#00dd88;border-bottom:1px solid #003322"><th>G</th><th>\u0394pos</th><th>\u0394vel</th><th>\u0394hdg</th></tr>';
for(const d of PAR.diff){h+='<tr style="border-bottom:1px solid rgba(0,100,60,.15)"><td>'+(PAR.diff.indexOf(d)+1)+'</td><td>'+d.d_pos.toFixed(4)+'</td><td>'+d.d_vel.toFixed(4)+'</td><td>'+d.d_hdg+'\u00b0</td></tr>'}
h+='</table></div></div></div>';
h+='<p style="font:9px sans-serif;color:#556;margin-top:6px">Purpose: The parallel scope measures trajectory sensitivity to barrel placement. The \u0394 at each gate = the correction needed if the barrel were shifted by the offset angle. Use (A-X+(A*X))*(+A-X)*(1/2)*(0/1) pattern: the product of sum/difference with the half-factor selects the corrective mode.</p>';
h+='</div>';
// SWARM
h+='<div class="cd"><h2 style="color:#cc44ff">\u2726 MULTI-BARREL SWARM \u2014 FREE PLACEMENT</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:8px">Each swarm barrel can be placed <b>anywhere</b> in AU space &mdash; not just on the scope ring. Each independently solves a transfer orbit to the SAME target. FTOP remains the orientation center; 7h is the canonical focus direction.</p>';
h+='<div class="rw"><span class="lb">Barrel clocks (snap)</span><input class="cfg-in" id="cfg-swmclk" type="text" style="width:180px" value="'+SWM.map(s=>s.ck).join(',')+'" placeholder="e.g. 1,3,5,9,11"></div>';
h+='<p style="font:8px sans-serif;color:#667;margin:2px 0 6px">Comma-separated clock positions for scope-ring snap. Or use X/Y inputs below for free placement.</p>';
// Per-barrel free placement table
h+='<div style="overflow-x:auto;margin-top:10px"><table style="width:100%;font-size:10px;border-collapse:collapse">';
h+='<tr style="color:#cc44ff;border-bottom:1px solid #331144"><th>#</th><th>Clock</th><th>X pos (AU)</th><th>Y pos (AU)</th><th>R(FTOP)</th><th>Angle</th><th>|v\u2080|</th><th>Miss</th><th>Status</th></tr>';
for(let si=0;si<SWM.length;si++){
const sw=SWM[si];const vm=Math.sqrt(sw.v0[0]*sw.v0[0]+sw.v0[1]*sw.v0[1]);
const sr=Math.sqrt(sw.bp[0]*sw.bp[0]+sw.bp[1]*sw.bp[1]);
const sang=Math.atan2(sw.bp[1],sw.bp[0])*180/Math.PI;
const ok=sw.miss<0.1;const col=SWM_COLS[si%SWM_COLS.length];
h+='<tr style="border-bottom:1px solid rgba(100,40,80,.2)">';
h+='<td style="color:'+col+'">'+( si+1)+'</td>';
h+='<td style="color:'+col+'">'+sw.ck+'h</td>';
h+='<td><input class="cfg-in" id="cfg-swx'+si+'" type="number" step="0.01" style="width:72px" value="'+sw.bp[0].toFixed(4)+'"></td>';
h+='<td><input class="cfg-in" id="cfg-swy'+si+'" type="number" step="0.01" style="width:72px" value="'+sw.bp[1].toFixed(4)+'"></td>';
h+='<td style="color:#889">'+sr.toFixed(3)+'</td>';
h+='<td style="color:#889">'+sang.toFixed(1)+'\u00b0</td>';
h+='<td style="color:'+col+'">'+vm.toFixed(3)+'</td>';
h+='<td>'+sw.miss+'</td>';
h+='<td style="color:'+(ok?'#44ff88':'#ff4444')+'">'+(ok?'\u2714':'\u2718')+'</td></tr>'}
h+='</table></div>';
h+='<p style="font:8px sans-serif;color:#667;margin-top:6px">Edit X/Y directly for free placement. Clock column is display-only after free placement. Edit the \u201cBarrel clocks\u201d field and hit APPLY to re-snap all barrels to scope ring positions.</p>';
h+='<p style="font:9px sans-serif;color:#556;margin-top:4px">N barrels \u2192 one target X. P(at least one hit) = 1 \u2212 \u220f(1\u2212p\u1d62). Free placement allows optimal aiming geometry for each barrel independently.</p>';
h+='</div>';
// DISPLAY OPTIONS
h+='<div class="cd"><h2>\ud83d\udd2d DISPLAY OPTIONS</h2>';
h+='<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">';
h+='<div class="rw"><span class="lb">Show cross-sections</span><input type="checkbox" id="cfg-xs" checked></div>';
h+='<div class="rw"><span class="lb">Show Fibonacci spiral</span><input type="checkbox" id="cfg-fib" checked></div>';
h+='<div class="rw"><span class="lb">Show velocity arrows</span><input type="checkbox" id="cfg-varr" checked></div>';
h+='<div class="rw"><span class="lb">Show tangent spheres</span><input type="checkbox" id="cfg-tsph" checked></div>';
h+='<div class="rw"><span class="lb">Show tensor mapping</span><input type="checkbox" id="cfg-tmap" checked></div>';
h+='<div class="rw"><span class="lb">Show midpoint annotations</span><input type="checkbox" id="cfg-mid" checked></div>';
h+='<div class="rw"><span class="lb">Show reference orbits</span><input type="checkbox" id="cfg-ref" checked></div>';
h+='<div class="rw"><span class="lb">Show HUD overlay</span><input type="checkbox" id="cfg-hud" checked></div>';
h+='<div class="rw"><span class="lb">Show AU ruler</span><input type="checkbox" id="cfg-ruler" checked></div>';
h+='</div></div>';
// ACTION BUTTONS
h+='<div class="cd" style="text-align:center"><h2>ACTIONS</h2>';
h+='<p style="font:10px sans-serif;color:#889;margin-bottom:12px">Applying changes requires re-simulation on the Python backend. Export current config or reset to defaults.</p>';
h+='<div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">';
h+='<button class="cfg-btn" id="cfg-apply" onclick="cfgApply()" style="background:#3a5a2a;color:#88ff88;border-color:#44ff88">\u25b6 APPLY & RE-SIMULATE</button>';
h+='<button class="cfg-btn" onclick="cfgExport()" style="background:#2a3a5a;color:#88bbff;border-color:#4488ff">\u21e9 EXPORT CONFIG JSON</button>';
h+='<button class="cfg-btn" onclick="cfgReset()" style="background:#5a3a2a;color:#ff8866;border-color:#ff6644">\u21ba RESET TO DEFAULTS</button>';
h+='</div></div>';
// STATUS
h+='<div class="cd"><h2>CURRENT SIMULATION STATUS</h2>';
h+='<div class="rw"><span class="lb">Last run</span><span class="vl">'+R.ts+'</span></div>';
h+='<div class="rw"><span class="lb">Baseline hit rate</span><span class="vl" style="color:#ff6644">'+R.bh+'%</span></div>';
h+='<div class="rw"><span class="lb">Corrected hit rate</span><span class="vl" style="color:#44ff88">'+R.ch+'%</span></div>';
h+='<div class="rw"><span class="lb">Mean miss (corrected)</span><span class="vl">'+R.cmm+' AU</span></div>';
h+='<div class="rw"><span class="lb">Cross-sections computed</span><span class="vl" style="color:#ffd700">'+GXY.length+' vesica piscis points</span></div>';
h+='<div class="rw"><span class="lb">Trajectory points</span><span class="vl" style="color:#ffd700">'+NT.length+' RK4 steps</span></div>';
h+='<div class="rw"><span class="lb">FoL sphere count</span><span class="vl">31 (7 inner + 12 scope + 12 outer)</span></div>';
h+='<div class="rw"><span class="lb">Pairwise tests</span><span class="vl">C(31,2) = 465</span></div>';
{const tm=TM;const trT=tm[0][0]+tm[1][1]+tm[2][2]+tm[3][3];
const frobT=Math.sqrt(tm.reduce((s,r)=>s+r.reduce((a,v)=>a+v*v,0),0));
const detT=(function(m){const a=m[0],b=m[1],c=m[2],d=m[3];return a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2])-b[2]*(c[1]*d[3]-c[3]*d[1])+b[3]*(c[1]*d[2]-c[2]*d[1]))-a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2])-b[2]*(c[0]*d[3]-c[3]*d[0])+b[3]*(c[0]*d[2]-c[2]*d[0]))+a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1])-b[1]*(c[0]*d[3]-c[3]*d[0])+b[3]*(c[0]*d[1]-c[1]*d[0]))-a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1])-b[1]*(c[0]*d[2]-c[2]*d[0])+b[2]*(c[0]*d[1]-c[1]*d[0]))})(tm);
h+='<div class="rw"><span class="lb">Tensor det(T)</span><span class="vl" style="color:#ffd700">'+detT.toFixed(6)+'</span></div>';
h+='<div class="rw"><span class="lb">Tensor tr(T)</span><span class="vl" style="color:#ffd700">'+trT.toFixed(6)+'</span></div>';
h+='<div class="rw"><span class="lb">Tensor ||T||_F</span><span class="vl" style="color:#ffd700">'+frobT.toFixed(4)+'</span></div>';}
h+='<div class="rw"><span class="lb">Sphere locations</span><span class="vl">5,184,000 (60\u00d760\u00d760\u00d724)</span></div>';
h+='</div>';
cp.innerHTML=h})();

// Wire projectile preset dropdowns — filling d/rho/Isp for each barrel row.
(function wireProjPresets(){const prefixes=['proj0','projP'];
for(let si=0;si<SWM.length;si++)prefixes.push('projS'+si);
for(const pr of prefixes){const sel=document.getElementById(pr+'_preset');if(!sel)continue;
sel.onchange=function(){const v=this.value;if(!v)return;
const[d,rho,isp]=v.split(',');
const dEl=document.getElementById(pr+'_d');const rEl=document.getElementById(pr+'_rho');const iEl=document.getElementById(pr+'_isp');
if(dEl)dEl.value=d;if(rEl)rEl.value=rho;if(iEl)iEl.value=isp}}})();

// Barrel live readout update (called on X/Y input)
function brlUpdate(){
const x=parseFloat(document.getElementById('cfg-bx').value)||0;
const y=parseFloat(document.getElementById('cfg-by').value)||0;
const r=Math.sqrt(x*x+y*y);
const ang=Math.atan2(y,x)*180/Math.PI;
const clk=Math.round(((90-ang+360)%360)/30)||12;
const rEl=document.getElementById('brl-r');
const aEl=document.getElementById('brl-ang');
const cEl=document.getElementById('brl-clk');
if(rEl)rEl.textContent=r.toFixed(4)+' AU';
if(aEl)aEl.textContent=ang.toFixed(1)+'\u00b0';
if(cEl)cEl.textContent=clk+'h'}
// Target live readout update (called on X/Y input)
function tgtUpdate(){
const x=parseFloat(document.getElementById('cfg-tx').value)||0;
const y=parseFloat(document.getElementById('cfg-ty').value)||0;
const r=Math.sqrt(x*x+y*y);
const ang=Math.atan2(y,x)*180/Math.PI;
const clk=Math.round(((90-ang+360)%360)/30)||12;
const rEl=document.getElementById('tgt-r');
const aEl=document.getElementById('tgt-ang');
const cEl=document.getElementById('tgt-clk');
const gEl=document.getElementById('tgt-grid');
if(rEl)rEl.textContent=r.toFixed(4)+' AU';
if(aEl)aEl.textContent=ang.toFixed(1)+'\u00b0';
if(cEl)cEl.textContent=clk+'h';
if(gEl)gEl.textContent='('+(x/Rs*30).toFixed(1)+', '+(y/Rs*30).toFixed(1)+')'}
// Snap primary barrel to clock position on scope ring
function snapBarrel(){
const clk=parseInt(document.getElementById('cfg-bclk').value)||7;
const ang=(90-(clk-1)*30)*Math.PI/180;
const x=Rs*Math.cos(ang),y=Rs*Math.sin(ang);
document.getElementById('cfg-bx').value=x.toFixed(4);
document.getElementById('cfg-by').value=y.toFixed(4);
brlUpdate()}
// Config apply — POST to /resim for live re-simulation
function cfgApply(){
const btn=document.getElementById('cfg-apply');
btn.textContent='\u23f3 SIMULATING...';btn.disabled=true;btn.style.opacity='0.5';
const swmStr=document.getElementById('cfg-swmclk')?document.getElementById('cfg-swmclk').value:'1,3,5,9,11';
const swmClks=swmStr.split(',').map(s=>parseInt(s.trim())).filter(n=>n>=1&&n<=12);
// Collect per-barrel free positions if set
const swmPos=[];for(let si=0;si<SWM.length;si++){
const xEl=document.getElementById('cfg-swx'+si);const yEl=document.getElementById('cfg-swy'+si);
if(xEl&&yEl)swmPos.push([parseFloat(xEl.value),parseFloat(yEl.value)]);
else swmPos.push(null)}
// Projectile spec (primary / default barrel knowledge). Prefer the list row
// proj0_* inputs when present, fall back to the older cfg-p* block.
const readProj=(prefix,fallback)=>{
const dEl=document.getElementById(prefix+'_d');const rEl=document.getElementById(prefix+'_rho');const iEl=document.getElementById(prefix+'_isp');
if(!dEl&&!rEl&&!iEl)return fallback;
const pf=v=>{const n=parseFloat(v);return isFinite(n)?n:null};
const out={diameter_m:dEl?pf(dEl.value):null,density_kg_m3:rEl?pf(rEl.value):null,Isp_s:iEl?pf(iEl.value):null};
// If all are null, treat as absent so the backend falls back to default.
if(out.diameter_m==null&&out.density_kg_m3==null&&out.Isp_s==null)return fallback;
return out};
const legacy=(function(){const pdEl=document.getElementById('cfg-pd'),phoEl=document.getElementById('cfg-prho'),pIspEl=document.getElementById('cfg-pisp');
return(pdEl&&phoEl&&pIspEl)?{diameter_m:parseFloat(pdEl.value),density_kg_m3:parseFloat(phoEl.value),Isp_s:parseFloat(pIspEl.value)}:null})();
const projSpec=readProj('proj0',legacy);
const parProj=readProj('projP',null);
const swmProj=[];for(let si=0;si<SWM.length;si++){swmProj.push(readProj('projS'+si,null))}
const tzEl=document.getElementById('cfg-tz');
const pertEl=document.getElementById('cfg-pert');
const body={bx:parseFloat(document.getElementById('cfg-bx').value),by:parseFloat(document.getElementById('cfg-by').value),
tx:parseFloat(document.getElementById('cfg-tx').value),ty:parseFloat(document.getElementById('cfg-ty').value),
tz:tzEl?parseFloat(tzEl.value):0,
hr:parseFloat(document.getElementById('cfg-hr').value),sims:parseInt(document.getElementById('cfg-ns').value),
tf:parseFloat(document.getElementById('cfg-tf').value),swarm_clocks:swmClks,swarm_positions:swmPos,
proj:projSpec,parallel_proj:parProj,swarm_proj:swmProj,
use_perturbations:pertEl?pertEl.checked:false};
fetch('/resim',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
.then(r=>r.json()).then(d=>{
if(d.ok){btn.textContent='\u2714 DONE — Reloading...';btn.style.background='#2a5a2a';setTimeout(()=>location.reload(),500)}
else{btn.textContent='\u2718 ERROR: '+d.msg;btn.style.background='#5a2a2a';btn.disabled=false;btn.style.opacity='1'}
}).catch(e=>{btn.textContent='\u2718 '+e.message;btn.style.background='#5a2a2a';btn.disabled=false;btn.style.opacity='1'})}
// Config export helper
function cfgExport(){const cfg={barrel:{x:parseFloat(document.getElementById("cfg-bx").value),y:parseFloat(document.getElementById("cfg-by").value),clock:parseInt(document.getElementById("cfg-bclk").value)},target:{x:parseFloat(document.getElementById("cfg-tx").value),y:parseFloat(document.getElementById("cfg-ty").value),hit_radius:parseFloat(document.getElementById("cfg-hr").value)},comet:{vx:parseFloat(document.getElementById("cfg-vx").value),vy:parseFloat(document.getElementById("cfg-vy").value),mass:parseFloat(document.getElementById("cfg-mass").value),diameter:parseFloat(document.getElementById("cfg-diam").value),type:document.getElementById("cfg-type").value,role:document.getElementById("cfg-role").value},flight:{tf:parseFloat(document.getElementById("cfg-tf").value),dt:parseFloat(document.getElementById("cfg-dt").value),mu:parseFloat(document.getElementById("cfg-mu").value),gates:parseInt(document.getElementById("cfg-ng").value),damping:parseFloat(document.getElementById("cfg-damp").value),max_dv:parseFloat(document.getElementById("cfg-maxdv").value),Rs:parseFloat(document.getElementById("cfg-rs").value)},mc:{n:parseInt(document.getElementById("cfg-ns").value),sigma_pos:parseFloat(document.getElementById("cfg-sigp").value),sigma_vel:parseFloat(document.getElementById("cfg-sigv").value),dist:document.getElementById("cfg-ndist").value,seed:parseInt(document.getElementById("cfg-seed").value)}};
const blob=new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='hit_config.json';a.click()}
function cfgReset(){document.getElementById("cfg-bx").value=B[0].toFixed(4);document.getElementById("cfg-by").value=B[1].toFixed(4);document.getElementById("cfg-tx").value=TG[0].toFixed(4);document.getElementById("cfg-ty").value=TG[1].toFixed(4);document.getElementById("cfg-vx").value=V0[0].toFixed(4);document.getElementById("cfg-vy").value=V0[1].toFixed(4);document.getElementById("cfg-tf").value=TF;document.getElementById("cfg-ns").value=R.n;alert('Reset to current simulation values.')}

ldStep(4,'ok');
window.addEventListener('resize',()=>requestAnimationFrame(draw));
// Safety: dismiss the loader after 8s no matter what, so a render glitch can
// never leave the user staring at a spinner.
setTimeout(ldDone,8000);
window.addEventListener('error',()=>ldDone(),{once:true});
requestAnimationFrame(()=>{try{draw()}finally{ldDone()}});
</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    _d=b'';_args=None
    def do_GET(self):
        self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length',str(len(self._d)))
        self.send_header('Cache-Control','no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma','no-cache');self.send_header('Expires','0')
        self.send_header('ETag',str(id(self._d)))
        self.end_headers();self.wfile.write(self._d)
    def do_POST(self):
        if self.path=='/resim':
            ln=int(self.headers.get('Content-Length',0))
            body=json.loads(self.rfile.read(ln)) if ln>0 else {}
            try:
                ns=int(body.get('sims',2000))
                bx=float(body.get('bx',B_DEF[0]));by=float(body.get('by',B_DEF[1]))
                tx=float(body.get('tx',TG_DEF[0]));ty=float(body.get('ty',TG_DEF[1]))
                hr=float(body.get('hr',0.02))
                tf=float(body.get('tf',0.70))
                swm_clks=body.get('swarm_clocks',[1,3,5,9,11])
                swm_pos=body.get('swarm_positions',None)  # list of [x,y] or null per barrel
                proj_in=body.get('proj',None)             # primary projectile: {diameter_m, density_kg_m3, Isp_s}
                par_proj=body.get('parallel_proj',None)   # parallel barrel override
                swm_proj=body.get('swarm_proj',None)      # list of per-barrel overrides (or null)
                sun_x=body.get('sun_x',None)              # None = Sun at FTOP
                sun_y=body.get('sun_y',None)
                use_pert=body.get('use_perturbations',False)
                # Target can be specified as either (tx, ty) OR as a planet/moon
                # selector (planet_idx, moon_idx) — the handler resolves the
                # selector against the current solar-system snapshot.
                planet_idx=body.get('planet_idx',None)
                moon_idx=body.get('moon_idx',None)
                tz=body.get('tz',None)      # optional z-component of target
                sun_z=body.get('sun_z',None)
                global HIT_RAD;HIT_RAD=hr
                S=System(ns=ns)
                if sun_x is not None and sun_y is not None:
                    S.sun_pos=np.array([float(sun_x),float(sun_y)])
                if sun_z is not None:
                    S.sun_z=float(sun_z)
                if tz is not None:
                    S.tz=float(tz)
                S.use_perturbations=bool(use_pert)
                # Resolve planet/moon target if requested (planet_idx=0 is the Sun
                # itself; >=1 is a planet in SOLAR; moon_idx indexes the planet's moon list).
                if planet_idx is not None and planet_idx!='':
                    pi=int(planet_idx)
                    if pi==0:
                        # Sun itself is the target — use its world-AU position
                        tx,ty=S.sun_pos[0],S.sun_pos[1]
                    elif 1<=pi<len(S.solar):
                        pb=S.solar[pi];th=pb['phase']
                        px=pb['r_orbit']*np.cos(th);py=pb['r_orbit']*np.sin(th)
                        if moon_idx is not None and moon_idx!='':
                            mi=int(moon_idx)
                            if 0<=mi<len(pb.get('moons',[])):
                                mb=pb['moons'][mi];mth=mb['phase']
                                px+=mb['r_orbit']*np.cos(mth)
                                py+=mb['r_orbit']*np.sin(mth)
                        tx,ty=float(S.sun_pos[0]+px),float(S.sun_pos[1]+py)
                S.bp=np.array([bx,by]);S.tp=np.array([tx,ty]);S.tf=tf
                S.dtg=S.tf/13.0;S.swm_clocks=swm_clks
                if swm_pos:S.swm_positions=[np.array(p) if p else None for p in swm_pos]
                # Re-run auto-grow now that the POST values have been applied.
                # The barrel is NOT re-snapped (user may have free-placed it);
                # but the scope ring + gxy get rebuilt so the template grows
                # uniformly around the new scene. "One larger scope, same
                # geometry, covering more AU" — no extra spheres.
                S._autogrow_Rs(snap_barrel=False,include_swarm=True)
                def _merge_proj(p):
                    m=dict(DEFAULT_PROJECTILE)
                    if p:m.update({k:p[k] for k in p if p[k] is not None})
                    return m
                if proj_in:S.proj=_merge_proj(proj_in)
                if par_proj:S.par_proj=_merge_proj(par_proj)
                if swm_proj:S.swm_projs=[_merge_proj(p) if p else None for p in swm_proj]
                S.v0=S._solve();S.gs,S.gJ=S._gates();S.T=S._tensor()
                r,bi,ci=S.run();viz=S.viz(r,bi,ci)
                H._d=HTML.replace('/*__DATA__*/null',json.dumps(viz)).encode('utf-8')
                resp=json.dumps({'ok':True,'msg':'Re-simulation complete. Refresh page.'})
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(resp)));self.end_headers()
                self.wfile.write(resp.encode())
            except Exception as e:
                resp=json.dumps({'ok':False,'msg':str(e)})
                self.send_response(500);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(resp)));self.end_headers()
                self.wfile.write(resp.encode())
        else:
            self.send_response(404);self.end_headers()
    def log_message(self,*a):pass

def main():
    import time
    ap=argparse.ArgumentParser(description='Hit.py — Tensor-Flower Comet Redirection')
    # Default sims dropped from 2000 → 300 so first launch takes ~2 s instead
    # of ~60 s. Crank it back up via --sims for publication-quality runs, or
    # via CONFIG → Monte-Carlo → N simulations inside the browser.
    ap.add_argument('--sims',type=int,default=300)
    ap.add_argument('--port',type=int,default=8080)
    ap.add_argument('--perturbations',action='store_true',
        help='Enable n-body planetary perturbations on initial run (slow). Default: off.')
    args=ap.parse_args()
    t0=time.time()
    def step(msg):
        dt=time.time()-t0
        print(f"  [{dt:5.2f}s] {msg}",flush=True)
    step("Constructing System + solving transfer orbit...")
    S=System(ns=args.sims)
    if args.perturbations:
        S.use_perturbations=True
        step("N-body perturbations ENABLED (expect a longer init)")
    step(f"Running Monte Carlo ({args.sims} sims × 2 campaigns)...")
    r,bi,ci=S.run()
    step("Computing parallel scope + swarm + viz payload...")
    viz=S.viz(r,bi,ci)
    step("Encoding JSON + HTML...")
    H._d=HTML.replace('/*__DATA__*/null',json.dumps(viz)).encode('utf-8')
    class T(http.server.HTTPServer):
        allow_reuse_address=True
        def process_request(self,rq,ad):threading.Thread(target=self.finish_request,args=(rq,ad),daemon=True).start()
    httpd=T(('127.0.0.1',args.port),H);threading.Thread(target=httpd.serve_forever,daemon=True).start()
    step(f"Server ready at http://127.0.0.1:{args.port}/")
    total=time.time()-t0
    print(f"\n  ◆ Dashboard ready in {total:.2f}s → http://127.0.0.1:{args.port}/")
    print("  Press Ctrl+C to stop.\n")
    webbrowser.open(f'http://127.0.0.1:{args.port}/')
    try:
        while True:threading.Event().wait(1)
    except KeyboardInterrupt:print("\n  Server stopped.")

if __name__=='__main__':main()
