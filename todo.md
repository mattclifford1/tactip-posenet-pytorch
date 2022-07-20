# List of Dev todos
CODE validation
- compare MAE of keras model to torch model (keras: https://data.bris.ac.uk/data/dataset/110f0tkyy28pa2joru2pxxbrxd)
  - keras data from above url:
    - edge_2d:    [2, 6]  [0.15, 1.45]     331: [0.15, 1.66]
    - surface_2d: [2, 6]  [0.08, 0.72]     331: [0.11, 0.73]
    
- write pytests!!!!

dataloader
(maybe change to just using the sim2real data?)
  - simpler image loading mechanism (remove weight transforms shite)


network
- change output size depending on 2d/3d
- flexible to let user define different conv channel sizes



validaiton
- better saving structure
- script to eval
  - give model dir and dataloader - return MAE


over-arching
- restructure repo into folders
- make into a pipeline
- add detailed readme
