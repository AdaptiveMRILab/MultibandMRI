MultibandMRI
============


MultibandMRI is a toolbox of k-space interpolation-based simultaneous
multi-slice (SMS) image reconstruction algorithms with harmonized PyTorch
implementation. The currently available algorithms include: 

1. Slice-GRAPPA[1]
2. Split-Slice-GRAPPA[2]
3. Readout-SENSE-GRAPPA[3]
4. Slice-RAKI[4]
5. Split-Slice-RAKI[5]
6. Readout-SENSE-RAKI[6]

Basic usage is shown below with more in-depth examples in ``sample_recon.ipynb``

.. code-block:: python 

    import MultibandMRI as mbmri

    ###############################################
    #     Calibrate/Train Interpolation Kernel    #
    ###############################################

    # GRAPPA reconstruction
    obj = mbmri.slice_grappa(
        data_cal,                # individual-slice calibration data (torch.complex64)
        accel=accel,             # length-two tuple with in-plane acceleration factors (always (1,R) for SMS)
        kernel_size=kernel_size, # length-two tuple with interpolation kernel sizes along row and column dims 
        tik=tik,                 # Tikhonov regularization parameter (float) 
        final_matrix_size=final_matrix_size # final interpolated in-plane image size 
    )

    # RAKI reconstruction 
    obj = mbmri.slice_raki(
        data_cal,                # individual-slice calibration data (torch.complex64) 
        raki_recon_folder,       # folder where trained network weights are saved  
        accel=accel,             # length-two tuple with in-plane acceleration factors (always (1,R) for SMS) 
        kernel_size=kernel_size, # length-two tuple with interpolation kernel sizes along row and column dims  
        final_matrix_size=final_matrix_size, # final interpolated in-plane image size 
        linear_weight=linear_weight,         # relative strength of linear reconstruction (0=RAKI, 1=residual RAKI)
        num_epochs=epochs,                   # number of epochs to fit network parameters 
        num_layers=nlayers,                  # (number of layers for MLP; number of residual blocks for RES)
        hidden_size=hsize,                   # hidden channels in linear units in neural networks
        train_split=train_split,             # training data fraction size 
        loss_function='L1_L2',               # 'L1', 'L2', or 'L1_L2'
        l2_frac=l2_frac,                     # strength of L2 component for 'L1_L2' loss
        net_type=net_type                    # ('MLP' = multi-layer perceptron, 'RES'= ResNet)
    )

    ###############################################
    #      Use Trained Kernel to Reconstruct      #
    ###############################################

    ksp_recon, _ = obj.apply(
        data_acc # slice-collapsed SMS data to be separated/interpolated (torch.complex64)
    )

Installation
------------

.. code:: bash

    git clone https://github.com/AdaptiveMRILab/MultibandMRI.git 
    cd MultibandMRI
    python3 -m venv .venv 
    source .venv/bin/activate 
    python -m pip install -r requirements.txt 
    python -m pip install -e .


References
------------

[1] Setsompop et al. Magn. Reson. Med. 67(5): 1210-1224 (2012).

[2] Cauley et al. Magn. Reson. Med. 72(1): 93-102 (2014). 

[3] Koopmans. Magn. Reson. Med. 77(3): 998-1009 (2017). 

[4] Mickevicius et al. Magn. Reson. Med. 84(2): 847-856 (2020).

[5] Nencka et al. Magn. Reson. Med. 85(6): 3272-3280 (2021). 

[6] Zhang et al. NeuroImage 256: 119248 (2022). 