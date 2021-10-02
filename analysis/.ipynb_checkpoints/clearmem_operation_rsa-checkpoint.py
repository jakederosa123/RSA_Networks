import clearmem_my_functions as mf
import os
import numpy as np
import nibabel as nib
import pandas as pd

'''
conda activate conda_fmri
xcluster='local'
xstep='ph1_create_nodes'
xnodes='glasser'
xsubj='001'
'''
'''
import imp
imp.reload(mf)
from clearmem_my_functions import *
'''

'''
#------ run all subjects
import clearmem_operation_rsa as cp
cp.clearmem_rsa('blanca','ph1_create_nodes','glasser', '001')
cp.clearmem_rsa('local','ph1_create_nodes','glasser', '001')
import imp
imp.reload(cp)
from clearmem_operation_rsa import *
'''


def clearmem_rsa(xcluster, xstep, xnodes, xsubj):
    # steps: True False
    steps = {'ph1_create_nodes': False, 'ph2_grp_results': False, 'ph3_kmean': False, xstep: True}

    ###########################################
    # arguments
    ###########################################
    xargs = {'cluster': xcluster,
             'subject_id': 'sub-%s' % xsubj,
             'nodes': xnodes,
             'node_mask': 'mask',
             'resolution': 1,
             'n_vox_cluster': 5}

    if xargs['nodes'] == 'glasser':
        xargs['n_nodes'] = 180

    ###########################################
    # logging
    ###########################################
    def logs(txt, xlog=False):
        print(txt)
        if xlog:
            # print >> xlog, txt
            print(txt, file=xlog)

    ###########################################
    # SET UP
    ###########################################
    dirs = mf.setup_directory(xargs)

    if steps['ph1_create_nodes']:
        xlog_dir = dirs['node_log']
    else:
        xlog_dir = dirs['operation_rsa_log']

    if os.path.isdir(xlog_dir) == False:
        os.mkdir(xlog_dir)

    ######### subjects
    f = open(os.path.join(dirs['params'], "subjectlist.txt"), "r")
    subject_lists = f.read().split('\n')
    f.close()
    subject_lists = list(filter(None, subject_lists))
    print('* Number of subjects: %s' % str(len(subject_lists)))

    xargs['subject_lists'] = subject_lists
    xargs['n_subjs'] = len(subject_lists)

    ######### Open log file
    xlog_fname = os.path.join(xlog_dir,
        'logs_%s_n%d.txt' % (xstep, xargs['n_subjs']))

    xlog = open(xlog_fname, 'a')
    logs(xargs, xlog)

    ###########################################
    # ph1_create_nodes: mask/load nii
    ###########################################
    if steps['ph1_create_nodes']:

        logs('#########################', xlog)
        logs('# PH1: preparation', xlog)
        logs('#########################', xlog)

        xparcel_prep = False
        xmni_node_prep = False
        xnode_mni2sub = True

        ######### pacels in MNI 1mm to subject func space
        # ------ redefine parcel id in MNI 1mm
        if xparcel_prep:
            mf.redefine_parcel(xargs, dirs, xlog, logs)
        
        if xmni_node_prep:
            mf.creat_parcel(xargs, dirs, xlog, logs)

        if xnode_mni2sub:
            mf.mni2sub_node(xargs, dirs, xlog, logs)

    ###########################################
    # ph2_grp_results: visualization
    ###########################################
    if steps['ph2_grp_results']:

        logs('################################', xlog)
        logs('# PH2: grp results visualization', xlog)
        logs('################################', xlog)

        import csv

        hemi_name = ['LH', 'RH']
        model_name = ['attention', 'inhibition', 'attention2']
        w_name = ['pat', 'pat-w']
        xd_name = ['p', 'n']  # positive, negative

        xparcel_file = os.path.join(dirs['parcels'],
                                    'ori_Parcels_MNI_%d.nii' % xargs['resolution'])
        xparcel = nib.load(xparcel_file)
        xparcel_mx = xparcel.get_fdata()

        xmap = {}

        for xhemi in hemi_name:
            for xm in model_name:
                for xw in w_name:
                    logs('(+) creating Nii for %s/ %s/ %s' % (xhemi, xm, xw))
                    # ------ read map
                    xmap_cvs = os.path.join(dirs['operation_rsa_grp_map'],
                                            'fit_bmap_fdr_%s_%s_%s.csv' % (xhemi, xm, xw))

                    xmean_p = []
                    xmean_n = []
                    xse_p = []
                    xse_n = []
                    xp_p = []
                    xp_n = []
                    xmean = []
                    xse = []
                    xp = []

                    with open(xmap_cvs) as csv_file:
                        csv_reader = csv.reader(csv_file, delimiter=',')
                        line_count = 0
                        for row in csv_reader:
                            if line_count == 0:
                                print(f'Column names are {", ".join(row)}')
                                line_count += 1
                            else:
                                line_count += 1
                                xmean.append(float(row[0]))
                                xse.append(float(row[1]))
                                xp.append(float(row[2]))

                                if float(row[0]) < 0:
                                    xmean_n.append(abs(float(row[0])))
                                    xse_n.append(abs(float(row[1])))
                                    xp_n.append(abs(float(row[2])))
                                else:
                                    xmean_p.append(float(row[0]))
                                    xse_p.append(float(row[1]))
                                    xp_p.append(float(row[2]))

                        print(f'Processed {line_count} lines.')

                    xmap['mean_%s_%s_%s_p' % (xhemi, xm, xw)] = xmean_p
                    xmap['mean_%s_%s_%s_n' % (xhemi, xm, xw)] = xmean_n
                    xmap['se_%s_%s_%s_p' % (xhemi, xm, xw)] = xse_p
                    xmap['se_%s_%s_%s_n' % (xhemi, xm, xw)] = xse_n
                    xmap['pvalue_%s_%s_%s_p' % (xhemi, xm, xw)] = xp_p
                    xmap['pvalue_%s_%s_%s_n' % (xhemi, xm, xw)] = xp_n

                    xmap['mean_%s_%s_%s' % (xhemi, xm, xw)] = xmean
                    xmap['se_%s_%s_%s' % (xhemi, xm, xw)] = xse
                    xmap['pvalue_%s_%s_%s' % (xhemi, xm, xw)] = xp

                    for xd in xd_name:
                        # ------ write on tmeplate
                        xbmap = np.zeros(xparcel.shape)
                        xpmap = np.zeros(xparcel.shape)

                        n_nodes = len(xmap['mean_%s_%s_%s_%s' % (xhemi, xm, xw, xd)])
                        if n_nodes > 0:
                            for xnode in range(n_nodes):
                                xnode_file = os.path.join(dirs['node_mni'],
                                                          'node_%s_%0.3d_1mm_%s_bin.nii.gz' % (
                                                          xargs['nodes'], xnode + 1, xhemi))
                                it_node = nib.load(xnode_file)
                                it_node_mx = it_node.get_fdata()

                                xbmap[np.where(it_node_mx == 1)] = xmap['mean_%s_%s_%s_%s' % (xhemi, xm, xw, xd)][xnode]
                                xpmap[np.where(it_node_mx == 1)] = xmap['pvalue_%s_%s_%s_%s' % (xhemi, xm, xw, xd)][
                                    xnode]

                            # ------ beta map
                            xbmap_img = nib.Nifti1Image(xbmap, affine=xparcel.affine)
                            xbmap_img.set_data_dtype(xparcel_mx.dtype)

                            xbimg_name = os.path.join(dirs['operation_rsa_grp_map'],
                                                      'bmap_fdr_%s_%s_%s_%s' % (xhemi, xm, xw, xd))
                            nib.save(xbmap_img, '%s.nii.gz' % xbimg_name)

                            # ------ pvalue map
                            xpmap_img = nib.Nifti1Image(xpmap, affine=xparcel.affine)
                            xpmap_img.set_data_dtype(xparcel_mx.dtype)

                            xpimg_name = os.path.join(dirs['operation_rsa_grp_map'],
                                                      'pmap_fdr_%s_%s_%s_%s' % (xhemi, xm, xw, xd))
                            nib.save(xpmap_img, '%s.nii.gz' % xpimg_name)

                    # ------ write on tmeplate
                    xbmap = np.zeros(xparcel.shape)
                    xpmap = np.zeros(xparcel.shape)

                    n_nodes = len(xmap['mean_%s_%s_%s' % (xhemi, xm, xw)])
                    if n_nodes > 0:
                        for xnode in range(n_nodes):
                            xnode_file = os.path.join(dirs['node_mni'],
                                                      'node_%s_%0.3d_1mm_%s_bin.nii.gz' % (
                                                      xargs['nodes'], xnode + 1, xhemi))
                            it_node = nib.load(xnode_file)
                            it_node_mx = it_node.get_fdata()

                            xbmap[np.where(it_node_mx == 1)] = xmap['mean_%s_%s_%s' % (xhemi, xm, xw)][xnode]
                            xpmap[np.where(it_node_mx == 1)] = xmap['pvalue_%s_%s_%s' % (xhemi, xm, xw)][xnode]

                        # ------ beta map
                        xbmap_img = nib.Nifti1Image(xbmap, affine=xparcel.affine)
                        xbmap_img.set_data_dtype(xparcel_mx.dtype)

                        xbimg_name = os.path.join(dirs['operation_rsa_grp_map'],
                                                  'bmap_fdr_%s_%s_%s' % (xhemi, xm, xw))
                        nib.save(xbmap_img, '%s.nii.gz' % xbimg_name)

                        # ------ pvalue map
                        xpmap_img = nib.Nifti1Image(xpmap, affine=xparcel.affine)
                        xpmap_img.set_data_dtype(xparcel_mx.dtype)
                        xpimg_name = os.path.join(dirs['operation_rsa_grp_map'],
                                                  'pmap_fdr_%s_%s_%s' % (xhemi, xm, xw))
                        nib.save(xpmap_img, '%s.nii.gz' % xpimg_name)

        ## combine RH/LH
        for xm in model_name:
            for xw in w_name:
                for xbp in ['bmap', 'pmap']:
                    xbimg_name_l = os.path.join(dirs['operation_rsa_grp_map'],
                                                '%s_fdr_LH_%s_%s' % (xbp, xm, xw))
                    xbimg_name_r = os.path.join(dirs['operation_rsa_grp_map'],
                                                '%s_fdr_RH_%s_%s' % (xbp, xm, xw))

                    xbimg_name = os.path.join(dirs['operation_rsa_grp_map'],
                                              '%s_fdr_%s_%s' % (xbp, xm, xw))

                    os.system('fslmaths %s -add %s %s' % (xbimg_name_l, xbimg_name_r, xbimg_name))

                    for xd in xd_name:
                        xbimg_name_l = os.path.join(dirs['operation_rsa_grp_map'],
                                                    '%s_fdr_LH_%s_%s_%s' % (xbp, xm, xw, xd))
                        xbimg_name_r = os.path.join(dirs['operation_rsa_grp_map'],
                                                    '%s_fdr_RH_%s_%s_%s' % (xbp, xm, xw, xd))

                        xbimg_name = os.path.join(dirs['operation_rsa_grp_map'],
                                                  '%s_fdr_%s_%s_%s' % (xbp, xm, xw, xd))

                        os.system('fslmaths %s -add %s %s' % (xbimg_name_l, xbimg_name_r, xbimg_name))

    ###########################################
    # ph3_kmean
    ###########################################
    if steps['ph3_kmean']:

        logs('################################', xlog)
        logs('# PH2: ph3_kmean', xlog)
        logs('################################', xlog)
        
        # parcel files
        xparcel_mx = {}
        for xhemi in ['LH','RH']:
            xparcel_file = os.path.join(dirs['parcels'], 'Parcels_MNI_1_%s.nii.gz' % xhemi)
            xparcel = nib.load(xparcel_file)
            xparcel_mx[xhemi] = xparcel.get_fdata()
        
        # mapping the idx
        alg_name = ['sqeuclidean','correlation']
        col_name = ['k_2', 'k_3', 'k_4', 'k_5']
        xkmean_id = {}
        for xa in alg_name:
            xcsv = os.path.join(dirs['operation_rsa_grp_kmean'], 'kmean_idx_%s.csv' % xa)
            xkmean_id[xa] = pd.read_csv(xcsv, index_col=None, header=0)
            
            n_nodes, n_clusters = xkmean_id[xa].shape
            xid_map = np.zeros(xparcel.shape)

            for xcl in range(0, n_clusters):
                for xnode in range(0, n_nodes):

                    if xnode < 180:
                        it_node = xnode + 1
                        xhemi = 'LH'
                    else:
                        it_node = xnode + 1 - 180
                        xhemi = 'RH'

                    xid_map[np.where(xparcel_mx[xhemi] == it_node)] = xkmean_id[xa].values[xnode, xcl]
                    xmap_img = nib.Nifti1Image(xid_map, affine=xparcel.affine)
                    xmap_img.set_data_dtype(xparcel_mx[xhemi].dtype)

                    xout_nii = os.path.join(dirs['operation_rsa_grp_kmean'], 
                        'kmean_idx_map_%s_cluster_%s' % (xa, col_name[xcl]))
                    nib.save(xmap_img, '%s.nii.gz' % xout_nii)

























