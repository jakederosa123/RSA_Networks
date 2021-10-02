#clearmem_my_functions.py
import os
import glob
import numpy as np
import nibabel as nib
import sys
from nipype.interfaces import fsl
import time

###########################################
# define directory
###########################################
#------ setup home directories
def setup_directory(xargs):
    dirs = {'data_home':''}

    if xargs['cluster']=='local':
        dirs_home = os.path.expanduser('~')
        dirs['data_home'] = os.path.join(dirs_home, 'Clearmem2/mvpa2')
        dirs['fsl_mni'] = '/usr/local/fsl/data/standard'
        dirs['src_home'] = os.path.join(dirs_home, 'Clearmem/imaging_data/utaustin')

    elif xargs['cluster']=='blanca':
        dirs['data_home'] = '/pl/active/banich/studies/wmem/fmri/mvpa2'
        dirs['fsl_mni'] = '/projects/ics/software/fsl/6.0.3/data/standard'
        dirs['src_home'] = '/pl/active/banich/studies/wmem/fmri/mvpa/utaustin/data'

    dirs['data_study'] = os.path.join(dirs['data_home'], 'data_study_fsl')
    dirs['params'] = os.path.join(dirs['data_home'], 'params')
    dirs['index'] = os.path.join(dirs['data_home'], 'index')
    dirs['spm'] = os.path.join(dirs['data_home'], 'spm')
    dirs['node_home'] = os.path.join(dirs['data_home'], 'node')
    dirs['node_log'] = os.path.join(dirs['node_home'], 'log')
    dirs['node'] = os.path.join(dirs['node_home'], xargs['nodes'])
    dirs['parcels'] = os.path.join(dirs['node'], 'parcels')
    dirs['node_subj'] = os.path.join(dirs['node'], 'subj_node')
    dirs['node_mni'] = os.path.join(dirs['node'], 'nodes_%dmm' % xargs['resolution'])

    #------ operation RSA
    dirs['operation_rsa'] = os.path.join(dirs['data_home'], 'operation_rsa')
    dirs['operation_rsa_log'] = os.path.join(dirs['operation_rsa'], 'log')
    dirs['operation_rsa_grp'] = os.path.join(dirs['operation_rsa'], 'grp')
    dirs['operation_rsa_grp_map'] = os.path.join(dirs['operation_rsa_grp'], 'map')
    dirs['operation_rsa_grp_kmean'] = os.path.join(dirs['operation_rsa_grp'], 'kmean')

    return dirs

###########################################
## redefine parcellation id
###########################################
def redefine_parcel(xargs, dirs, xlog, logs):
    ## checking the L/R MNI masks
    
    logs('(+) redefine parcel ids: %s' % xargs['nodes'], xlog)

    xtmpdir = os.path.join(dirs['parcels'], 'tmp')
    if (os.path.isdir(xtmpdir)==False):
        os.mkdir(xtmpdir)

    xparcel_file = os.path.join(dirs['parcels'], 
        'ori_Parcels_MNI_%d.nii' % xargs['resolution']) 
    xparcel = nib.load(xparcel_file)
    xhdr = xparcel.header
    xparcel_mx = xparcel.get_fdata()
    xparcel_mx = np.round(xparcel_mx)
    n_pars = int(np.max(xparcel_mx))
    aff = xparcel.affine
    xcent = nib.affines.apply_affine(npl.inv(aff), [0, 0, 0])[0]

    npar = 0
    for xv in range(n_pars):
        logs('* pacel %0.3d' % (xv+1), xlog)
        xdata = np.zeros(xparcel.shape)
        xdata[np.where(xparcel_mx==(xv+1))]=1
        xpar_img = nib.Nifti1Image(xdata, affine=xparcel.affine)
        xpar_img.set_data_dtype(xparcel_mx.dtype)
        
        xvox = np.array(np.where(xparcel_mx==(xv+1)))
        xpar = 'tmp_node_%s_%0.3d_%dmm' % (xargs['nodes'], xv+1, xargs['resolution'])
        xpar_file = os.path.join(xtmpdir, xpar)
        xpar_bin = '%s_bin' % xpar_file

        nib.save(xpar_img, '%s.nii.gz' % xpar_file)
        os.system('fslmaths %s -bin %s' % (xpar_file, xpar_bin))
        os.remove('%s.nii.gz' % xpar_file)

        ## split to LH/RH 
        xparcel_new = nib.load('%s.nii.gz' % xpar_bin)
        xparcel_new_mx = xparcel_new.get_fdata()
        xvox_bin = np.array(np.where(xparcel_new_mx==1))

        n_new_vox = 0
        if (np.min(xvox_bin[0]) < xcent) and (np.max(xvox_bin[0]) > xcent):
            ###########################################
            # left hemisphere
            npar = npar+1
            xtmp_new_l = os.path.join(xtmpdir, 
                'tmp_l_node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            os.system('fslmaths %s -bin -roi %d %d 0 -1 0 -1 0 -1 %s' 
                % (xpar_bin, xcent, xcent*2, xtmp_new_l))
            
            ## save it as the parcel id
            xhemi_par = nib.load('%s.nii.gz' % xtmp_new_l)
            xhemi_par_mx = xhemi_par.get_fdata()
            xdata = np.zeros(xhemi_par.shape)
            xdata[np.where(xhemi_par_mx==1)]=npar
            xpar_img = nib.Nifti1Image(xdata, affine=xhemi_par.affine)
            xpar_img.set_data_dtype(xhemi_par_mx.dtype)

            xnew_l = os.path.join(xtmpdir, 
                'node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            nib.save(xpar_img, '%s.nii.gz' % xnew_l)

            xvox_new = np.array(np.where(xdata==npar))
            n_new_vox += xvox_new.shape[1]
            logs('... changing intensity of %d voxels to the LH parcel id %0.3d' 
                % (xvox_new.shape[1], npar), xlog)

            ###########################################
            # right hemisphere
            npar = npar+1
            xtmp_new_r = os.path.join(xtmpdir, 
                'tmp_r_node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            os.system('fslmaths %s -sub %s %s' 
                % (xpar_bin, xnew_l, xtmp_new_r))
            
            ## save it as the parcel id
            xhemi_par = nib.load('%s.nii.gz' % xtmp_new_r)
            xhemi_par_mx = xhemi_par.get_fdata()
            xdata = np.zeros(xhemi_par.shape)
            xdata[np.where(xhemi_par_mx==1)]=npar
            xpar_img = nib.Nifti1Image(xdata, affine=xhemi_par.affine)
            xpar_img.set_data_dtype(xhemi_par_mx.dtype)

            xnew_r = os.path.join(xtmpdir, 
                'node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            nib.save(xpar_img, '%s.nii.gz' % xnew_r)
            os.remove('%s.nii.gz' % xtmp_new_l)
            os.remove('%s.nii.gz' % xtmp_new_r)
            os.remove('%s.nii.gz' % xpar_bin)

            xvox_new = np.array(np.where(xdata==npar))
            n_new_vox += xvox_new.shape[1]
            logs('... changing intensity of %d voxels to the RH parcel id %0.3d' 
                % (xvox_new.shape[1], npar), xlog)
        else:
            if (np.max(xvox_bin[0]) > xcent):
                logs('... pacel only in LH')
                xhemi = 'LH'
            elif (np.min(xvox_bin[0]) < xcent):
                logs('... pacel only in RH')
                xhemi = 'RH'
            
            ###########################################
            # hemisphere
            npar = npar+1
            xtmp_new_h = os.path.join(xtmpdir, 
                'tmp_h_node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            os.system('mv %s.nii.gz %s.nii.gz' % (xpar_bin, xtmp_new_h))
            
            ## save it as the parcel id
            xhemi_par = nib.load('%s.nii.gz' % xtmp_new_h)
            xhemi_par_mx = xhemi_par.get_fdata()
            xdata = np.zeros(xhemi_par.shape)
            xdata[np.where(xhemi_par_mx==1)]=npar
            xpar_img = nib.Nifti1Image(xdata, affine=xhemi_par.affine)
            xpar_img.set_data_dtype(xhemi_par_mx.dtype)

            xnew_h = os.path.join(xtmpdir, 
                'node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            nib.save(xpar_img, '%s.nii.gz' % xnew_h)
            os.remove('%s.nii.gz' % xtmp_new_h)

            xvox_new = np.array(np.where(xdata==npar))
            n_new_vox += xvox_new.shape[1]
            logs('... changing intensity of %d voxels to the %s parcel id %0.3d' 
                % (xvox_new.shape[1], xhemi, npar), xlog)

        logs('... changing %d voxels (0.5mm) to %d voxels (%dmm)' 
            % (xvox_bin.shape[1], n_new_vox, xargs['resolution']), xlog)

    xpars = glob.glob(os.path.join(xtmpdir, 'node_%s*%smm.nii.gz' % (xargs['nodes'], xargs['resolution'])))
    xpars_all = os.path.join(dirs['parcels'], 'Parcels_MNI_%s' % xargs['resolution'])

    for xv in range(len(xpars)):
        if xv==0:
            os.system('cp %s %s.nii.gz' % (xpars[xv], xpars_all))
        else:
            os.system('fslmaths %s -add %s %s' % (xpars_all, xpars[xv], xpars_all))

    os.rmdir(xtmpdir)
    os.system('gunzip %s.nii.gz' % xpars_all)

###########################################
## create nodes from the parcellation mask
###########################################
def creat_parcel(xargs, dirs, xlog, logs):
    
    logs('(+) create nodes for %s parcellation' % xargs['nodes'], xlog)

    start = time.time()

    xmask_file = os.path.join(dirs['fsl_mni'], 'MNI152_T1_%dmm_brain_mask.nii.gz' % xargs['resolution'])
    xoutdir = dirs['node_mni']
    xparcel_file = os.path.join(dirs['parcels'], 'Parcels_MNI_%d.nii.gz' % xargs['resolution'])
    xparcel = nib.load(xparcel_file)
    xhdr = xparcel.header
    xparcel_mx = xparcel.get_fdata()
    n_pars = int(np.max(xparcel_mx))
    aff = xparcel.affine
    xcent = nib.affines.apply_affine(npl.inv(aff), [0, 0, 0])[0]

    logs('* create roi from %d parcels defined in %s' % (n_pars, xparcel_file), xlog)
    logs('... center x coordinates: %s' % xcent, xlog)

    nl = 0
    nr = 0
    roi_selection = {}
    it_rois_nl = []
    it_rois_nr = []
    n_vox_in_sphere_nl = []
    n_vox_in_sphere_nr = []
    n_novox_rois_nl = []
    n_novox_rois_nr = []

    for xv in range(n_pars):
        logs('... parcel: %d' % (xv+1), xlog)
        xdata = np.zeros(xparcel.shape)
        xdata[np.where(xparcel_mx==(xv+1))]=1
        xpar_img = nib.Nifti1Image(xdata, affine=xparcel.affine)
        xpar_img.set_data_dtype(xparcel_mx.dtype)
        
        if np.max(np.where(xdata==1)[0]) < xcent:
            xhemi = 'RH'
            nl = nl+1
            xpar = 'node_%s_%0.3d_%dmm_%s' % (xargs['nodes'], nl, xargs['resolution'], xhemi)
        else:
            xhemi = 'LH'
            nr = nr+1
            xpar = 'node_%s_%0.3d_%dmm_%s' % (xargs['nodes'], nr, xargs['resolution'], xhemi)

        xvox = np.array(np.where(xparcel_mx==(xv+1)))
        logs('..... %d voxel was found for %s before filtering' % (xvox.shape[1], xpar), xlog)

        xpar_file = os.path.join(xoutdir, xpar)
        nib.save(xpar_img, xpar_file)

        os.system('fslmaths %s -bin %s_bin' % (xpar_file, xpar_file))
        os.system('fslmaths %s_bin -mas %s %s_bin' % (xpar_file, xmask_file, xpar_file))
        os.remove('%s.nii' % xpar_file)

        ## check if the mask is empty
        xtmp_par = nib.load('%s_bin.nii.gz' % xpar_file)
        xtmp_par_mx = xtmp_par.get_fdata()
        xvox_filtered = np.array(np.where(xtmp_par_mx==1))
        xmax = xvox_filtered.shape[1]

        if xmax<xargs['n_vox_cluster']:
            logs('..... no voxel was found for %s' % xpar, xlog)
            if xhemi=='RH':
                n_novox_rois_nl.append(nl)
            else:
                n_novox_rois_nr.append(nr)

            os.remove('%s_bin.nii.gz' % xpar_file)
        else:
            logs('..... %d voxel was found for %s after filtering' % (xvox_filtered.shape[1], xpar), xlog)
            if xhemi=='RH':
                it_rois_nl.append(nl)
                n_vox_in_sphere_nl.append(xvox_filtered.shape[1])
            else:
                it_rois_nr.append(nr)
                n_vox_in_sphere_nr.append(xvox_filtered.shape[1])

            # os.system('gunzip %s_bin.nii.gz' % xpar_file)
    
    roi_selection['n_all_pars'] = n_pars
    roi_selection['n_rois_LH'] = nl
    roi_selection['n_rois_RH'] = nr
    roi_selection['n_sel_rois_LH'] = len(it_rois_nl)
    roi_selection['n_sel_rois_RH'] = len(it_rois_nr)
    roi_selection['it_rois_LH'] = it_rois_nl
    roi_selection['it_rois_RH'] = it_rois_nr
    roi_selection['n_vox_in_sphere_LH'] = n_vox_in_sphere_nl
    roi_selection['n_vox_in_sphere_RH'] = n_vox_in_sphere_nr
    roi_selection['n_novox_rois_LH'] = n_novox_rois_nl
    roi_selection['n_novox_rois_RH'] = n_novox_rois_nr

    outfile = os.path.join(xoutdir, 'rois_%s_%dmm.npy' 
        % (xargs['node_mask'], xargs['resolution']))
    np.save(outfile, roi_selection)

    ###########################################
    ## change resolution of glasser parcels
    ###########################################
    ## separate all nodes > change resolution > binary

    ## create collapsing pars to varify
    for xhemi in ['LH','RH']:
        all_pars = os.path.join(xoutdir, 'all_pars_MNI_%dmm_%s.nii.gz' 
            % (xargs['resolution'], xhemi))
        xfiles = glob.glob(os.path.join(xoutdir, 'node*%s*' % xhemi))

        for xv in range(len(xfiles)):
            xpar_file = xfiles[xv]
            if xv==0:
                os.system('cp %s %s' % (xpar_file, all_pars))
            else:
                os.system('fslmaths %s -add %s %s' % (all_pars, xpar_file, all_pars))

    end = time.time()
    total_time = end - start
    m, s = divmod(total_time, 60)
    logs('Time elapsed: %s minutes %s seconds' % (round(m), round(s)), xlog)

###########################################
## convert MNI nodes to subject nodes
###########################################
def mni2sub_node(xargs, dirs, xlog, logs):

    start = time.time()

    logs('(+) convert MNI nodes to subject nodes: %s' % xargs['subject_id'], xlog)

    # link inverse transform (standard to MPRAGE)
    mni2sub = os.path.join(dirs['src_home'], 'clearmem_v1_sub%s' 
        % xargs['subject_id'][-3:],'bold','avg_func_ref','mni2sub.nii.gz')
    mni2t1 = os.path.join(dirs['data_study'], xargs['subject_id'], 
        '%s_mni2t1.nii.gz' % xargs['subject_id'])

    if not os.path.isfile(mni2t1):
        os.system('ln -s %s %s' % (mni2sub, mni2t1))

    # link the func to MPRAGE mat file 
    sub2ref = os.path.join(dirs['src_home'], 'clearmem_v1_sub%s' 
        % xargs['subject_id'][-3:],'bold','avg_func_ref','sub2ref')
    t12func = os.path.join(dirs['data_study'], xargs['subject_id'], 
        '%s_t12func' % xargs['subject_id'])
    
    if not os.path.isfile(t12func):
        os.system('ln -s %s %s' % (sub2ref, t12func))        

    ###########################################
    # node directory
    func_mask = os.path.join(dirs['data_study'], xargs['subject_id'], 
        '%s_brain_mask.nii.gz' % xargs['subject_id'])
    xsubnode = os.path.join(dirs['node_subj'], xargs['subject_id'])
    xspm_mask = os.path.join(dirs['spm'], 'spm_%s' 
        % xargs['subject_id'], '%s.nii' % xargs['node_mask'])

    if (os.path.isdir(xsubnode)==False):
        os.mkdir(xsubnode)

    roi_selection = {}
    for xhemi in ['LH', 'RH']:
        node_list = glob.glob(os.path.join(dirs['node_mni'], 
            'node*%s*' % xhemi))
        n_pars = len(node_list)
        
        logs('* check roi from %d %s parcels in %s' % (n_pars, xargs['nodes'], xhemi), xlog)

        n_vox_in_node = []
        it_rois = []
        n_sel_vox_in_node = []

        for xv in range(n_pars):
            logs('* parcel: %0.3d' % (xv+1), xlog)

            xmni_node = os.path.join(dirs['node_mni'], 
                'node_%s_%0.3d_%dmm_%s_bin.nii.gz' 
                % (xargs['nodes'], (xv+1), xargs['resolution'], xhemi))
            xnode = 'node_%s_%s_%0.3d_%s' % (xargs['nodes'], xargs['subject_id'], (xv+1), xhemi)
            xsub_node = os.path.join(xsubnode, xnode)
            xsub_node_bin = '%s_bin' % xsub_node
            xsub_node_mas = '%s_mas' % xsub_node

            ####### applywarp
            logs('... applywarp to %s' % xnode, xlog)
            aw = fsl.ApplyWarp()
            aw.inputs.ref_file = func_mask
            aw.inputs.in_file = xmni_node
            aw.inputs.field_file = mni2t1
            aw.inputs.postmat = t12func
            aw.inputs.out_file = '%s.nii.gz' % xsub_node
            # logs('... %s' % aw.cmdline, xlog)
            res = aw.run() 

            ####### binarized version
            logs('... binarized to %s_bin' % xnode, xlog)
            os.system('fslmaths %s.nii.gz -bin %s.nii.gz' % (xsub_node, xsub_node_bin))
            os.remove('%s.nii.gz' % xsub_node)

            #------ voxel nubmer
            xori_parcel = nib.load('%s.nii.gz' % xsub_node_bin)
            xori_parcel_mx = xori_parcel.get_fdata()
            xunit = np.where(xori_parcel_mx>0)
            n_vox_in_node.append(len(xunit[0]))
            logs('..... %d voxels before filtering' % len(xunit[0]), xlog)
            
            ####### filter with spm_masl
            logs('... masked by spm mask > %s_mas' % xnode, xlog)
            os.system('fslmaths %s.nii.gz -mas %s %s.nii.gz' 
                % (xsub_node_bin, xspm_mask, xsub_node_mas))

            #------ voxel nubmer
            xparcel = nib.load('%s.nii.gz' % xsub_node_mas)
            xparcel_mx = xparcel.get_fdata()
            xunit = np.where(xparcel_mx>0)
            
            #------ adding all parcels
            if xv==0:
                xdata = np.zeros(xparcel.shape)
                logs('..... dimension: %s' % str(xparcel.shape), xlog)

            if len(xunit[0])>0:
                xdata[xunit]=xdata[xunit]+1
                n_sel_vox_in_node.append(len(xunit[0]))
                it_rois.append(xv+1)
                logs('..... %d voxels after filtering' % len(xunit[0]), xlog)
            else:
                logs('..... no voxels after filtering', xlog)
                os.remove('%s.nii.gz' % xsub_node_mas)

        os.remove('%s.nii.gz' % xsub_node_bin)
        xpar_file = os.path.join(xsubnode, 
            'all_pars_%s_%s' % (xargs['subject_id'], xhemi))
        xpar_img = nib.Nifti1Image(xdata, affine=xparcel.affine)
        xpar_img.set_data_dtype(xparcel_mx.dtype)
        nib.save(xpar_img, xpar_file)
        
        roi_selection['n_pars_%s' % xhemi] = n_pars
        roi_selection['n_vox_in_node_%s' % xhemi] = n_vox_in_node
        roi_selection['n_sel_vox_in_node_%s' % xhemi] = n_sel_vox_in_node
        roi_selection['it_rois_%s' % xhemi] = it_rois

    outfile = os.path.join(xsubnode, 'rois_%s_%s.npy' 
        % (xargs['nodes'], xargs['subject_id']))
    np.save(outfile, roi_selection)

    end = time.time()
    total_time = end - start
    m, s = divmod(total_time, 60)
    logs('Time elapsed: %s minutes %s seconds' % (round(m), round(s)), xlog)


