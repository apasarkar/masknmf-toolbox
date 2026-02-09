Taking another pass at Churchland lab WFCI data, documenting what has been done here. 
Date: Jan 21 2025

What did I do: 
1. Download two of Chris Kasniak's datasets (these are imaging.raw.mov files - same as Joao). It looks like they've been processed with the same infrastructure etc.
2. Run motion correction. It's important that the data is stored in a fast access drive (in my case, the 8TB drive led to faster motion correction).
3. Run the baseline code (wfield_cli._baseline).
     - All of the above stuff is done in explore_chris_kasniak_data.ipynb.
4. Test/update the dataloaders. Looks like it works well.
5. Figure out how the masking works. This time we make two standard deviation images, one on the blood channel and another on the gcamp channel, and we do a logical_or operationt to fuse these into one final mask.
6. version_controlled in a dedicated branch
7. Run hemocorrection pipeline on this data


Conventions for running the pipeline script: 
- We assume that the folder containing the motion corrected .bin stack is the same folder that contains the mask. Furthermore, the mask file is named manual_mask.npy.
- The bin_folder parameter for this script is the folder containing the .bin file (and the mask)
- Note the larger block size values (100 x 100) here


Things I learned/things to note: 
1. Chris Krasniak data has functional channel = 1 (not 0). The way to check this is as follows: 


times = one.load_dataset(eid_used, 'imaging.times.npy')
channels = one.load_dataset(eid_used, 'imaging.imagingLightSource.npy')
channel_info = one.load_dataset(eid_used, 'imagingLightSource.properties.htsv', download_only=True)
<!-- channel_info = pd.read_csv(channel_info, sep='\t') -->
# channel_info = pd.read_csv(channel_info, sep=',')
channel_info = pd.read_csv(
    channel_info,
    sep=r'[\t,]',
    engine='python'
)


# If haemocorrected need to take timestamps that correspond to functional channel
functional_channel = 470
functional_chn = channel_info.loc[channel_info['wavelength'] == functional_channel]['channel_id'].values[0]

## Decide whether data[:, 0, :, :] is the functional channel or data[:, 1, :, :]
functional_chn_index = 0 if channels[0] == functional_chn else 1
print(functional_chn_index)

NOTE: the above code reads the csv based in the case where entries are comma or backslash separated 

3. The wfield repo compression normalizes frames from each channel, subtracting a channel mean (frames_avg[0, :, :] or frames_avg[1, :, :]) and dividing by that same mean. NOTE: IN PREVIOUS ITERATIONS OF CODE, WE ONYL USED frame_avg[0,:, :] FOR BOTH CHANNELS TO UNNORMALIZE. THIS IS TECHNICALLY WRONG. We will moving forward represent these kinds of arrays as PMD objects ("factorized arrays") to avoid dealing with boilerplate code.
4. 


Next steps (Feb 6th)
1.  Make a visualizer comparing the gcamp, blood, hemocorr data from both pipelines
       - In "channel_by_channel_compare.ipyn"
2. Make a GUI showing how the trial-averaged signal differs between methods. DONE, in "hemo_compare_trial_avg"