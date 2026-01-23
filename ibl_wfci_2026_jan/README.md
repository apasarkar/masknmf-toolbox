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

Next Tasks:
1. Make a new hemocorrection pipeline and run it on this data. 