import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import exp, ceil, floor, sin, cos, tan, asin, acos, atan2, pi, e


def random_texture(W, H, activity=0.2, show=False):
    tx = np.random.random((W,H))
    tx[np.where(tx > 1-activity)] = 1
    tx[np.where(tx <= 1-activity)] = 0
    tx = np.array(tx, dtype=int)
    if show:
        plt.imshow(tx, cmap='binary')
        plt.colorbar()
        plt.show()
    return tx


def smooth_keyframe(evts, speed):
    impulse = int(1/speed)
    evts_smooth = np.zeros_like(evts, dtype=int)
    evts_smooth[np.where(evts != 0)] = impulse
    for i in range(1, evts.shape[0]):
        evts_smooth[i,:,:] = np.clip(evts_smooth[i-1,:,:]-1, 0, impulse)
        evts_smooth[np.where(evts != 0)] = impulse
    return evts_smooth


def animate(i):
    for ax, frame, title in zip(axs, frames, titles):
        ax.cla()
        ax.imshow(frame[i,:,:], cmap='binary', vmin=0, vmax=np.max(frame))
        ax.set_title(title)
        ax.axis('off')


def suppress(evts_in, evts_inb, RFW, RFH, CFW, CFH):
    evts_out = np.array(evts_in)

    evts_bg = np.array(evts_inb)
    evts_bg[:,:,int((RFW-CFW)/2):RFW-int((RFW-CFW)/2),int((RFH-CFH)/2):RFH-int((RFH-CFH)/2)] = 0
    evts_fg = np.zeros_like(evts_inb)
    evts_fg[:,:,int((RFW-CFW)/2):RFW-int((RFW-CFW)/2),int((RFH-CFH)/2):RFH-int((RFH-CFH)/2)] = evts_inb[:,:,int((RFW-CFW)/2):RFW-int((RFW-CFW)/2),int((RFH-CFH)/2):RFH-int((RFH-CFH)/2)]

    dir_bg = list(); dir_fg = list()
    for i in range(4):
        dir_bg.append(np.sum(evts_bg[:,i,:,:]))
        dir_fg.append(np.sum(evts_fg[:,i,:,:]))
    ptheta_bg = atan2(dir_bg[1]-dir_bg[3],dir_bg[0]-dir_bg[2])
    ptheta_fg = atan2(dir_fg[1]-dir_fg[3],dir_fg[0]-dir_fg[2])
    print(f'BG direction is {180*ptheta_bg/pi} deg. FG direction is {180*ptheta_fg/pi} deg.')
    if ptheta_fg > ptheta_bg - pi/10 and ptheta_fg < ptheta_bg + pi/10:
        print(f'\tSuppress!')
        evts_out[:,int((RFW-CFW)/2):RFW-int((RFW-CFW)/2),int((RFH-CFH)/2):RFH-int((RFH-CFH)/2)] = 0
    return evts_out


W = H = 100
RFW = RFH = 21
CFW = CFH = 11
v_bg = 0.05; theta_bg = pi*60/180; vx_bg = v_bg*cos(theta_bg); vy_bg = v_bg*sin(theta_bg)
v_fg = 0.1; theta_fg = pi*240/180; vx_fg = v_fg*cos(theta_fg); vy_fg = v_fg*sin(theta_fg)
print(f'BG speed {round(v_bg,3)}, heading {round(180*theta_bg/pi)%360} deg. vx = {round(vx_bg,3)}, vy = {round(vy_bg,3)}.')
print(f'FG speed {round(v_fg,3)}, heading {round(180*theta_fg/pi)%360} deg. vx = {round(vx_fg,3)}, vy = {round(vy_fg,3)}.')

T_sim = 1000

# Generate background motion by scrolling a random texture
tx_bg = random_texture(W, H)
evts_in = np.zeros((T_sim, RFW, RFH))
x = 0; y = 0
for t in range(T_sim):
    if floor(abs(x)) != floor(abs(x+vx_bg)) or floor(abs(y)) != floor(abs(y+vy_bg)):
        if floor(abs(x)) != floor(abs(x+vx_bg)):
            if vx_bg > 0:
                tx_bg = np.roll(tx_bg, 1, 0)
            else:
                tx_bg = np.roll(tx_bg, -1, 0)
        if floor(abs(y)) != floor(abs(y+vy_bg)):
            if vy_bg > 0:
                tx_bg = np.roll(tx_bg, 1, 1)
            else:
                tx_bg = np.roll(tx_bg, -1, 1)
        evts_in[t,:,:] = tx_bg[:RFW,:RFH]
    x += vx_bg; y += vy_bg

# Replace events from background with foreground motion in the center
evts_in[:,int((RFW-CFW)/2):RFW-int((RFW-CFW)/2),int((RFH-CFH)/2):RFH-int((RFH-CFH)/2)] = 0
tx_fg = random_texture(W, H, activity=0.4)
for t in range(T_sim):
    if floor(abs(x)) != floor(abs(x+vx_fg)) or floor(abs(y)) != floor(abs(y+vy_fg)):
        if floor(abs(x)) != floor(abs(x+vx_bg)):
            if vx_fg > 0:
                tx_fg = np.roll(tx_fg, 1, 0)
            else:
                tx_fg = np.roll(tx_fg, -1, 0)
        if floor(abs(y)) != floor(abs(y+vy_fg)):
            if vy_fg > 0:
                tx_fg = np.roll(tx_fg, 1, 1)
            else:
                tx_fg = np.roll(tx_fg, -1, 1)
        evts_in[t,int((RFW-CFW)/2):RFW-int((RFW-CFW)/2),int((RFH-CFH)/2):RFH-int((RFH-CFH)/2)] = tx_fg[:CFW,:CFH]
    x += vx_fg; y += vy_fg

# Simulation
I = 10; Vhi = 0.9*2*I; Vlo = 0.9*I*(1+e)/e
tauA = 1/v_bg
vAs = np.zeros((T_sim, 4, RFW, RFH))
vs = np.zeros((T_sim, 4, RFW, RFH))
evts_inb = np.zeros_like(vs)
for t in range(T_sim):
    vAs[t,0,1:,:] = vAs[t-1,0,1:,:]  + I*evts_in[t,:-1,:]
    vAs[t,1,:,1:] = vAs[t-1,1,:,1:] + I*evts_in[t,:,:-1]
    vAs[t,2,:-1,:] = vAs[t-1,2,:-1,:] + I*evts_in[t,1:,:]
    vAs[t,3,:,:-1] = vAs[t-1,3,:,:-1] + I*evts_in[t,:,1:]
    vAs = np.clip(vAs, 0, I)

    for i in range(4):
        vs[t,i,:,:] = vAs[t,i,:,:] + I*evts_in[t,:,:]
        evts_inb[t,i,*np.where(np.logical_and(vs[t,i,:,:] > Vlo, vs[t,i,:,:] < Vhi))] = 1

    for i in range(4):
        vAs[t,i,:,:] += -1*vAs[t,i,:,:]/tauA
    vAs = np.clip(vAs, 0, I)


evts_suppressed = suppress(evts_in, evts_inb, RFW, RFH, CFW, CFH)

fig, axs = plt.subplots(nrows=1, ncols=6)
frames = [
    smooth_keyframe(evts_in, v_bg),
    smooth_keyframe(evts_inb[:,0,:,:], v_bg),
    smooth_keyframe(evts_inb[:,1,:,:], v_bg),
    smooth_keyframe(evts_inb[:,2,:,:], v_bg),
    smooth_keyframe(evts_inb[:,3,:,:], v_bg),
    smooth_keyframe(evts_suppressed, v_bg)
]
titles = [
    'input events',
    'east-bound',
    'north-bound',
    'west-bound',
    'south-bound',
    'post-suppressed',
]
anim = animation.FuncAnimation(fig, animate, frames=T_sim, interval=1, blit=False)
plt.show()
