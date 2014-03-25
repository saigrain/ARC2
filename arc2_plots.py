import numpy as np
import pylab as plt
import os, os.path, string
import scipy.io
import glob

from SuzPyUtils.norm import *
from SuzPyUtils.multiplot import *

dataroot = '/Users/aigrain/Data/Kepler/KepSys/'
figroot = '/Users/aigrain/Documents/Publications/InPrep/ARC2/'

# def sel_nb(quarter = 2, mod = 2, out = 1, nnb = 8, \
#            root_dir = root_default, verbose = True, \
#            do_plot = True):
#     '''
#     Go through the CBV corrections with different numbers of basis functions
#     and select the best version depending on the ratio of range and
#     point-to-point scatter compared to raw and PDC light curves
#     '''
#     # Read in data files
#     rawdata = scipy.io.loadmat('%s/Q%d/q%d_mod%02d_out%d_raw.mat' % \
#                                (root_dir, quarter, quarter, mod, out))
#     time = np.array(rawdata['time']).flatten()        
#     t0 = int(np.floor(mymin(time)))
#     time -= t0
#     kid = rawdata['kid_arr'].flatten()
#     raw_flux = rawdata['flux_arr']
#     pdc_flux = rawdata['flux_arr_pdc']
#     crowd = rawdata['crowd'].flatten()
#     flfrc = rawdata['flfrc'].flatten()
#     nobj, nobs = raw_flux.shape
#     nbs = np.arange(nnb) + 1
#     cbv_flux = np.zeros((nnb, nobj, nobs))
#     for i in np.arange(nnb):
#         file = '%s/Q%d/q%d_mod%02d_out%d_cbv_%d.mat' % \
#             (root_dir, quarter, quarter, mod, out, i+1)
#         nbs[i] = int(string.split(os.path.splitext(file)[0], '_')[-1])
#         corrdata = scipy.io.loadmat(file)
#         cbv_flux[i,:,:] = corrdata['flux_cbv']
#     # Prepare arrays to store results
#     med_raw = np.zeros(nobj)
#     sig_raw = np.zeros(nobj)
#     sig_pdc = np.zeros(nobj)
#     sig_cbv = np.zeros((nnb,nobj))
#     ran_raw = np.zeros(nobj)
#     ran_pdc = np.zeros(nobj)
#     ran_cbv = np.zeros((nnb,nobj))
#     nb_opt = np.zeros(nobj, 'int')
#     sig_opt = np.zeros(nobj)
#     ran_opt = np.zeros(nobj)
#     flux_opt = np.zeros((nobj, nobs))
#     flag1 = np.zeros(nobj, 'bool')
#     flag2 = np.zeros(nobj, 'bool')
#     flag3 = np.zeros(nobj, 'bool')
#     flag4 = np.zeros(nobj, 'bool')
#     # Run through the light curves one by one
#     st0 = '---i/nobj ------KID ---median --sig-raw --sig-pdc --sig_cbv --ran-raw --ran-pdc --ran_cbv nb flag'
#     for i in np.arange(nobj):
#         if verbose == True:
#             if i % 20 == 0:
#                 print st0
#         # Apply contamination and flux-loss corrections to SAP data
#         raw = raw_flux[i,:]
#         l = np.isfinite(raw)
#         if np.isfinite(crowd[i]) * (crowd[i] != 0.0):
#             m = np.median(raw[l])
#             raw = raw - (1 - crowd[i]) * m 
#         if np.isfinite(flfrc[i]) * (flfrc[i] != 0.0):
#             raw = raw / flfrc[i] 
#         # Measure range annd p2p scatter in SAP data
#         med_raw[i] = np.median(raw[l])
#         raw = raw / med_raw[i]
#         raws = np.sort(raw)
#         ng = l.sum()
#         ran_raw[i] = raws[int(ng*0.95)] - raws[int(ng*0.05)]
#         diff = raw[1:] - raw[:-1]
#         l = np.isfinite(diff)
#         sig_raw[i] = 1.48 * np.median(np.abs(diff[l]))
#         # Measure range annd p2p scatter in PDC data
#         pdc = pdc_flux[i,:]
#         l = np.isfinite(pdc)
#         med = np.median(pdc[l])
#         pdc = pdc / med_raw[i]
#         pdcs = np.sort(pdc)
#         ng = l.sum()
#         ran_pdc[i] = pdcs[int(ng*0.95)] - pdcs[int(ng*0.05)]
#         diff = pdc[1:] - pdc[:-1]
#         l = np.isfinite(diff)
#         sig_pdc[i] = 1.48 * np.median(np.abs(diff[l]))
#         # Apply contamination and flux-loss corrections to CBV data,
#         # then measure range annd p2p scatter
#         for j in np.arange(nnb):
#             cbv = cbv_flux[j,i,:]
#             l = np.isfinite(cbv)
#             med = np.median(cbv[l])
#             if np.isfinite(crowd[i]) * (crowd[i] != 0.0):
#                 cbv = cbv - (1 - crowd[i]) * med 
#             if np.isfinite(flfrc[i]) * (flfrc[i] != 0.0):
#                 cbv = cbv / flfrc[i] 
#             cbv = cbv / med_raw[i]
#             cbvs = np.sort(cbv)
#             ng = l.sum()
#             ran_cbv[j,i] = cbvs[int(ng*0.95)] - cbvs[int(ng*0.05)]
#             diff = cbv[1:] - cbv[:-1]
#             l = np.isfinite(diff)
#             sig_cbv[j,i] = 1.48 * np.median(np.abs(diff[l]))
#         # Select the best number of basis functions
#         # (smallest number that significantly reduces range)
#         rc = ran_cbv[:,i]
#         med = np.median(rc)
#         sig = 1.48 * np.median(abs(rc - med))
#         jj = np.where(rc < med + 3 * sig)[0][0]
#         # Does that introduce noise? If so try to reduce nB till it doesn't
#         if sig_cbv[jj,i] > 1.1 * sig_raw[i]:
#             while jj > 0:
#                 jj -= 1
#                 if sig_cbv[jj,i] <= 1.1 * sig_raw[i]: break
#         nb_opt[i] = nbs[jj]
#         sig_opt[i] = sig_cbv[jj,i]
#         ran_opt[i] = ran_cbv[jj,i]
#         flux_opt[i,:] = cbv_flux[jj,i,:]
#         # Ok, how do we compare to SAP and PDC now?
#         if sig_opt[i] > 1.1 * sig_raw[i]: flag1[i] = True # CBV worse than SAP
#         if sig_opt[i] > 1.1 * sig_pdc[i]: flag2[i] = True # CBV worse than PDC
#         if sig_pdc[i] > 1.1 * sig_raw[i]: flag3[i] = True # PDC worse than SAP
#         if sig_pdc[i] > 1.1 * sig_opt[i]: flag4[i] = True # PDC worse than CBV
#         if verbose == True:
#             st = '%4i/%4i %9d %9d %9.7f %9.7f %9.7f %9.7f %9.7f %9.7f %2d %1d%1d%1d%1d' % \
#                 (i, nobj, kid[i], med_raw[i], sig_raw[i], \
#                 sig_pdc[i], sig_opt[i], ran_raw[i], \
#                 ran_pdc[i], ran_opt[i], nb_opt[i], flag1[i], flag2[i], flag3[i], flag4[i])
#             print st
#         if do_plot == True:
#             plt.figure(1)
#             plt.clf()
#             ax1 = plt.subplot(211)
#             plt.axhline(1.0, ls = '-', c = 'grey')
#             plt.axhline(1.1, ls = '--', c = 'grey')
#             plt.plot(nbs, sig_cbv[:,i].flatten()/sig_raw[i], 'b.', mec = 'b')
#             plt.plot(nb_opt[i], sig_opt[i]/sig_raw[i], 'r.', mec = 'r')
#             plt.plot(1, sig_pdc[i]/sig_raw[i], 'k.')
#             plt.ylabel('$\sigma/\sigma_0$')
#             ymax = max(sig_pdc[i]/sig_raw[i], (sig_cbv[:,i].flatten()/sig_raw[i]).max())
#             ymin = min(sig_pdc[i]/sig_raw[i], (sig_cbv[:,i].flatten()/sig_raw[i]).min())
#             yr = ymax - ymin
#             plt.ylim(ymin - 0.1 * ymin, ymax + 0.1 * ymin)
#             plt.subplot(212, sharex = ax1)
#             plt.axhline(med/ran_raw[i], ls = '-', c = 'grey')
#             plt.axhline((med+3*sig)/ran_raw[i], ls = '--', c = 'grey')
#             plt.plot(nbs, ran_cbv[:,i].flatten()/ran_raw[i], 'b.', mec = 'b')
#             plt.plot(nb_opt[i], ran_cbv[jj,i]/ran_raw[i], 'r.', mec = 'r')
#             plt.plot(1, ran_pdc[i]/ran_raw[i], 'k.')            
#             plt.ylabel('$R/R_0$')
#             plt.xlim(0, nbs.max() + 1)
#             plt.xlabel('nB')
#             ymax = max(ran_pdc[i]/ran_raw[i], (ran_cbv[:,i].flatten()/ran_raw[i]).max())
#             ymin = min(ran_pdc[i]/ran_raw[i], (ran_cbv[:,i].flatten()/ran_raw[i]).min())
#             yr = ymax - ymin
#             plt.ylim(ymin - 0.1 * ymin, ymax + 0.1 * ymin)
#             plt.figure(2)
#             plt.clf()
#             plt.plot(time, raw, '-', c = 'grey')
#             plt.plot(time, pdc - 10 * sig_raw[i], 'k-')
#             cols = ['c','m','y']
#             for j in np.arange(3):
#                 jjj = j + jj - 1
#                 if jjj < 0: continue
#                 if jjj > nnb-1: continue
#                 cbv = cbv_flux[jjj,i,:]
#                 l = np.isfinite(cbv)
#                 med = np.median(cbv[l])
#                 cbv = cbv / med
#                 plt.plot(time, cbv - (10 + 10 * (j+1)) * sig_raw[i], '-', c = cols[j])
#             plt.xlabel('time')
#             raw_input()           
#     scipy.io.savemat('%s/Q%d/q%d_mod%02d_out%d_cbv_sel.mat' % \
#                      (root_dir, quarter, quarter, mod, out), \
#                      {'flux_cbv': flux_opt, 'flag1': flag1, \
#                      'flag2': flag2, 'flag3': flag3, 'flag4': flag4})
#     scipy.io.savemat('%s/Q%d/q%d_mod%02d_out%d_cbv_stats.mat' % \
#                      (root_dir, quarter, quarter, mod, out), \
#                      {'nbs': nbs, 'kid': kid, 'med_raw': med_raw, \
#                       'sig_raw': sig_raw, 'ran_raw': ran_raw, \
#                       'sig_pdc': sig_pdc, 'ran_pdc': ran_pdc, \
#                       'sig_cbv': sig_cbv, 'ran_cbv': ran_cbv, \
#                       'sig_opt': sig_opt, 'ran_opt': ran_opt, \
#                       'nb_opt': nb_opt, 'flag1': flag1, \
#                      'flag2': flag2, 'flag3': flag3, 'flag4': flag4})
#     return

def plot_examples(quarter = 3, mod = 7, out = 3, mode = 'random', \
                  curves = None, nex = 5):
    # read raw data
    rawdata = scipy.io.loadmat('%sQ%d/q%d_mod%02d_out%d_raw.mat' % \
                               (dataroot, quarter, quarter, mod, out))
    time = np.array(rawdata['time']).flatten()        
    t0 = int(np.floor(mymin(time)))
    time -= t0
    kids = rawdata['kid_arr'].flatten()
    raws = rawdata['flux_arr']
    pdcs = rawdata['flux_arr_pdc']
    crowd = rawdata['crowd'].flatten()
    flfrc = rawdata['flfrc'].flatten()
    # select objects to plot
    nobj, nobs = raws.shape
    if mode == 'bright':
        data = scipy.io.loadmat('%s/Q%d/q%d_mod%02d_out%d_cbv_stats.mat' % \
                                (dataroot, quarter, quarter, mod, out))
        med_raw = data['med_raw'].flatten()
        s = np.argsort(med_raw)
        curves = s[-nex:]
        suff = '_bright'
    elif mode == 'faint':
        data = scipy.io.loadmat('%s/Q%d/q%d_mod%02d_out%d_cbv_stats.mat' % \
                                (dataroot, quarter, quarter, mod, out))
        med_raw = data['med_raw'].flatten()
        s = np.argsort(med_raw)
        curves = s[:nex]
        suff = '_faint'
    elif mode == 'bad':
        data = scipy.io.loadmat('%s/Q%d/q%d_mod%02d_out%d_cbv_stats.mat' % \
                                (dataroot, quarter, quarter, mod, out))
        flag = data['flag1'].flatten()
        curves = np.where(flag == True)[0]
        nex = min(10, len(curves))
        suff = '_bad'
    elif mode == 'better_than_pdc':
        data = scipy.io.loadmat('%s/Q%d/q%d_mod%02d_out%d_cbv_stats.mat' % \
                                (dataroot, quarter, quarter, mod, out))
        flag = data['flag3'].flatten()
        curves = np.where(flag == True)[0]
        nex = len(curves)
        suff = '_better_than_pdc'
    elif mode == 'random':
        curves = np.floor(np.random.rand(nex) * nobj).astype('int')    
        suff = '_rand'
    else:
        if curves == None:
            print  'Error: must specify either mode or curves'
            return
        nex = len(curves)
        suff = ''
    # read corrected data
    corrdata = scipy.io.loadmat('%sQ%d/q%d_mod%02d_out%d_cbv_sel.mat' % \
                                (dataroot, quarter, quarter, mod, out))
    cbvs = corrdata['flux_cbv']
    flag1 = corrdata['flag1'].flatten()
    flag2 = corrdata['flag2'].flatten()
    flag3 = corrdata['flag3'].flatten()
    flag4 = corrdata['flag4'].flatten()
    # plot objects
    ee1 = dofig(1, 1, nex)
    ee2 = dofig(2, 1, nex)
    for i in np.arange(nex):
        curve = curves[i]
        kid = kids[curve]
        print 'mod %02d out %d curve %04d KID %09d CBV' % (mod, out, curve, kid)
        raw_flux = raws[curve,:].flatten()
        pdc_flux = pdcs[curve,:].flatten() 
        cbv_flux = cbvs[curve,:].flatten()
        crowd_c = crowd[curve]
        flfrc_c = flfrc[curve]
        # Apply contamination and flux-loss corrections to SAP data
        if np.isfinite(crowd_c) * (crowd_c != 0.0):
            l = np.isfinite(raw_flux)
            m = np.median(raw_flux[l])
            raw_flux = raw_flux - (1 - crowd_c) * m 
        if np.isfinite(flfrc_c) * (flfrc_c != 0.0):
            raw_flux = raw_flux / flfrc_c 
        m = mymean(raw_flux)
        raw_flux = raw_flux / m - 1
        pdc_flux = pdcs[curve,:].flatten() 
        pdc_flux = pdc_flux / mymean(pdc_flux) - 1
        pdc_corr = raw_flux - pdc_flux
        cbv_flux = cbvs[curve,:].flatten() 
        # Apply contamination and flux-loss corrections to SAP data
        if np.isfinite(crowd_c) * (crowd_c != 0.0):
            l = np.isfinite(cbv_flux)
            m = np.median(cbv_flux[l])
            cbv_flux = cbv_flux - (1 - crowd_c) * m 
        if np.isfinite(flfrc_c) * (flfrc_c != 0.0):
            cbv_flux = cbv_flux / flfrc_c 
        cbv_flux = cbv_flux / mymean(cbv_flux) - 1
        cbv_corr = raw_flux - cbv_flux
        plt.figure(1)
        if i == 0:
            ax11 = doaxes(ee1, 1, nex, 0, 0, extra = 0)
        else:
            axc1 = doaxes(ee1, 1, nex, 0, i, sharex = ax11, extra = 0)
        plt.plot(time, raw_flux, '-', c = 'grey')
        offset = 5 * mystd(pdc_flux)
        yoff = offset
        plt.plot(time, pdc_corr - yoff, 'c-')
        yoff += offset
        plt.plot(time, cbv_corr - yoff, 'm-')
        ymin = mymin(cbv_corr)-yoff
        ymax = mymax(raw_flux)
        l = scipy.isfinite(pdc_corr)
        if pdc_corr[l][-1] > pdc_corr[l][0]:
            plt.text(time[0] + 0.95 * (time[-1]-time[0]), ymin, \
                       'KID %d' % kid, horizontalalignment = 'right', \
                       verticalalignment = 'bottom')
        else:
            plt.text(time[0] + 0.95 * (time[-1]-time[0]), ymax, \
                       'KID %d' % kid, horizontalalignment = 'right', \
                       verticalalignment = 'top')
        yr = ymax - ymin
        plt.ylim(ymin - 0.05 * yr, ymax + 0.1 * yr)
        plt.xlim(mymin(time), mymax(time))
        plt.figure(2)
        if i == 0:
            ax12 = doaxes(ee2, 1, nex, 0, 0, extra = 0)
        else:
            axc2 = doaxes(ee2, 1, nex, 0, i, sharex = ax12, extra = 0)
        plt.plot(time, pdc_flux, 'c-')
        yoff = offset
        plt.plot(time, cbv_flux - yoff, 'm-')
        ymin = mymin(cbv_flux - yoff)
        ymax = mymax(pdc_flux)
        yr = ymax - ymin
        plt.ylim(ymin - 0.05 * yr, ymax + 0.05 * yr)
        plt.xlim(mymin(time), mymax(time))
    plt.figure(1)
    plt.xlabel('time (BJD - %d)' % (2454833 + t0))
    plt.figure(2)
    plt.xlabel('time (BJD - %d)' % (2454833 + t0))
    return

def mk_fig1():
    # plot_examples(quarter=6,mod=7,out=3,curves=[1132,132,2356,1441,1458],mode='sel')
    plot_examples(quarter=3,mod=17,out=2,curves=[879,1844,541,1147],mode='sel')
    plt.figure(1)
    plt.savefig('%sfig1a.png' % figroot)
    plt.figure(2)
    plt.savefig('%sfig1b.png' % figroot)
    return

def sig_ran_comp(quarter = [3,3,3,3], mod = [2,7,13,17], out = [1,3,1,2]):
    nq = len(quarter)
    col = ['grey','c','m','y']
    lin = ['-','--',':','-.']
    for i in np.arange(nq):
        data = scipy.io.loadmat('%s/Q%d/q%d_mod%02d_out%d_cbv_stats.mat' % \
                                (dataroot, quarter[i], quarter[i], mod[i], out[i]))
        if i == 0:
            med_raw = data['med_raw'].flatten()
            sig_raw = data['sig_raw'].flatten()
            sig_pdc = data['sig_pdc'].flatten()
            sig_cbv = data['sig_opt'].flatten()
            ran_raw = data['ran_raw'].flatten()
            ran_pdc = data['ran_pdc'].flatten()
            ran_cbv = data['ran_opt'].flatten()
            nobj = len(med_raw)
            index = np.zeros(nobj, 'int')
            lab = np.array(['%d.%d' % (mod[i], out[i])])
        else:
            tmp = data['med_raw'].flatten()
            med_raw = np.append(med_raw, tmp)
            sig_raw = np.append(sig_raw, data['sig_raw'].flatten())
            sig_pdc = np.append(sig_pdc, data['sig_pdc'].flatten())
            sig_cbv = np.append(sig_cbv, data['sig_opt'].flatten())
            ran_raw = np.append(ran_raw, data['ran_raw'].flatten())
            ran_pdc = np.append(ran_pdc, data['ran_pdc'].flatten())
            ran_cbv = np.append(ran_cbv, data['ran_opt'].flatten())
            nobj = len(tmp)
            index = np.append(index, np.zeros(nobj, 'int') + i)
            lab = np.append(lab, '%d.%d' % (mod[i], out[i]))
    nobj = len(med_raw)
    mag = 21.0 - 2.5 * np.log10(med_raw)

    ee = dofig(1, 1, 3, aspect = 1.)
    ax1 = doaxes(ee, 1, 3, 0, 0)
    plt.axhline(1.0, ls = '-', c = 'grey')
    for i in np.arange(nq):
        l = index == i
        plt.plot(mag[l], sig_pdc[l] / sig_raw[l], '.', c = col[i], mec = col[i])
    plt.ylabel('$\sigma_{\mathrm{PDC}} / \sigma_{\mathrm{SAP}}$')
    ax2 = doaxes(ee, 1, 3, 0, 1, sharex = ax1, sharey = ax1)
    plt.axhline(1.0, ls = '-', c = 'grey')
    for i in np.arange(nq):
        l = index == i
        plt.plot(mag[l], sig_cbv[l] / sig_raw[l], '.', c = col[i], mec = col[i], label = lab[i])
    plt.legend(loc = 0)
    plt.ylabel('$\sigma_{\mathrm{CBV}} / \sigma_{\mathrm{SAP}}$')
    ax3 = doaxes(ee, 1, 3, 0, 2, sharex = ax1, sharey = ax1)
    plt.axhline(1.0, ls = '-', c = 'grey')
    for i in np.arange(nq):
        l = index == i
        plt.plot(mag[l], sig_pdc[l] / sig_cbv[l], '.', c = col[i], mec = col[i])
    print np.median(sig_pdc / sig_cbv)
    l = sig_pdc > 1.1 * sig_cbv
    print len(l), l.sum()
    l = sig_pdc < 0.9 * sig_cbv
    print len(l), l.sum()
    
    plt.ylabel('$\sigma_{\mathrm{PDC}} / \sigma_{\mathrm{CBV}}$')
    plt.ylim(0.8,1.5)
    plt.xlabel('$21 - 2.5 \log(\overline{F})$')
    plt.xlim(2,13)
    
    ee = dofig(2, 1, 3, aspect = 1.)
    ax1 = doaxes(ee, 1, 3, 0, 0)
    plt.axhline(1.0, ls = '-', c = 'grey')
    for i in np.arange(nq):
        l = index == i
        plt.plot(mag[l], ran_pdc[l] / ran_raw[l], '.', c = col[i], mec = col[i])
    plt.ylabel('$R_{\mathrm{PDC}}/R_{\mathrm{SAP}}$')
    ax2 = doaxes(ee, 1, 3, 0, 1, sharex = ax1, sharey = ax1)
    plt.axhline(1.0, ls = '-', c = 'grey')
    for i in np.arange(nq):
        l = index == i
        plt.plot(mag[l], ran_cbv[l] / ran_raw[l], '.', c = col[i], mec = col[i])
    plt.ylabel('$R_{\mathrm{CBV}}/R_{\mathrm{SAP}}$')
    ax3 = doaxes(ee, 1, 3, 0, 2, sharex = ax1, sharey = ax1)
    plt.axhline(1.0, ls = '-', c = 'grey')
    for i in np.arange(nq):
        l = index == i
        plt.plot(mag[l], ran_pdc[l] / ran_cbv[l], '.', c = col[i], mec = col[i])
    print np.median(ran_pdc / ran_cbv)
    l = ran_pdc > 1.1 * ran_cbv
    print len(l), l.sum()
    l = ran_pdc < 0.9 * ran_cbv
    print len(l), l.sum()
    plt.ylabel('$R_{\mathrm{PDC}}/R_{\mathrm{CBV}}$')
    plt.ylim(0,2)
    plt.xlabel('$21 - 2.5 \log(\overline{F})$')
    plt.xlim(2,13)
    return

def mk_fig2():
    sig_ran_comp(quarter=[3,3,3,3],mod=[2,7,13,17],out=[1,3,1,2])
    plt.figure(1)
    plt.savefig('%s/fig2a.png' % figroot)
    plt.figure(2)
    plt.savefig('%s/fig2b.png' % figroot)
    return
    
    # plt.close(4)
    # plt.figure(4)
    # for i in np.arange(nq):
    #     l = index == i
    #     x = np.sort(nb_opt[l])
    #     y = np.arange(len(x))/float(len(x))
    #     plt.plot(x, y, ls = lin[i], c = col[i])
    # plt.xlabel('No. CBVs')
    # ply.ylim(0,1)

    # cols = ['b','c','m','r','#FF9900','y','#99FF33','g']
