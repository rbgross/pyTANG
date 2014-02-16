# TANG: Comparing output pose data with ground-truth for virtual videos
# Run script: source("scripts/analyze-pose.r")

# Libraries
library(ggplot2)
library(grid)
library(gridExtra)
library(orientlib)  # for converting between Euler angles and quaternions
library(dtw)

# Script parameters
## Files and directories
fcurves_dir = "res/videos/virtual/"
default_pose_dir = "out/pose/"
all_test_sets = c("01-translate-X", "02-translate-Y", "03-translate-Z", "04-rotate-X", "05-rotate-Y", "06-rotate-Z")
fps_rates = c(10, 20, 30, 40, 50)  # for multi-speed pose comparisons
doPlotGraphs = FALSE

## Data scaling and transformation
fcurves_trans_scale = 100  # fcurves are in meters
pose_trans_scale = 0.5  # observed pose data is roughly in 0.5 cms (TODO verify this)

## Misc. constants
two_pi = 2 * pi

# Pose plotting (and analysis) function
plotPose <- function(fcurves_file, pose_file, pose_dir=default_pose_dir) {
  # * Read fcurves and prepare ground truth data
  fcurves = read.table(paste(fcurves_dir, fcurves_file, sep=""), header=TRUE, sep="\t")
  ground_truth = as.data.frame(list(frame=fcurves[,'frame']))  # seed ground truth data frame with 'frame' column
  
  # ** Transform and normalize fcurve translation values
  # NOTE For fcurves, Y & Z are flipped: y = -location_1, z = -location_2)
  ground_truth$true_trans_x = fcurves$location_0 * fcurves_trans_scale
  ground_truth$true_trans_y = -fcurves$location_1 * fcurves_trans_scale
  ground_truth$true_trans_z = -fcurves$location_2 * fcurves_trans_scale
  
  # ** Transform and normalize fcurve rotation values (Euler angles) to [-pi,+pi] range
  ground_truth$true_rot_x = (fcurves$rotation_euler_0 + (3 * pi / 2)) %% two_pi - pi  # requires a +pi/2 because of Y-Z axis change, and then normalization by: (x + pi) mod 2*pi - pi
  ground_truth$true_rot_y = -((fcurves$rotation_euler_1 + pi) %% two_pi - pi)  # needs to be flipped (negative)
  ground_truth$true_rot_z = -((fcurves$rotation_euler_2 + pi) %% two_pi - pi)  # needs to be flipped (negative)
  
  # ** Convert fcurve rotation angles to quaternions
  true_eulers = eulerzyx(as.matrix(ground_truth[, c('true_rot_z', 'true_rot_y', 'true_rot_x')]))
  true_quats = quaternion(true_eulers)
  #print(true_quats)
  true_quats_df = as.data.frame(true_quats[[]])
  colnames(true_quats_df) <- c('true_q1', 'true_q2', 'true_q3', 'true_q4')  # prevent name clash when merging with pose data
  ground_truth = cbind(ground_truth, true_quats_df)
  
  # * Read observed pose data
  pose = read.table(paste(pose_dir, pose_file, sep=""), header=TRUE, sep="\t")  # relative to camera
  
  # ** Normalize observed pose data
  pose$trans_x = pose$trans_x * pose_trans_scale
  pose$trans_y = pose$trans_y * pose_trans_scale
  pose$trans_z = pose$trans_z * pose_trans_scale
  
  # ** Convert pose rotation angles to quaternions
  pose_eulers = eulerzyx(as.matrix(pose[, c('rot_z', 'rot_y', 'rot_x')]))
  pose_quats = quaternion(pose_eulers)
  #print(pose_quats)
  pose_quats_df = as.data.frame(pose_quats[[]])
  pose = cbind(pose, pose_quats_df)
  
  # * Join datasets by frame number
  merged = merge(ground_truth, pose)
  #print(merged)
  
  if(doPlotGraphs) {
    # * Create merged plots
    mergedTransPlot = ggplot(data=merged, aes(x=frame)) +
      geom_line(aes(y=true_trans_x, colour="trans_x"), linetype="dashed") +
      geom_line(aes(y=true_trans_y, colour="trans_y"), linetype="dashed") +
      geom_line(aes(y=true_trans_z, colour="trans_z"), linetype="dashed") +
      geom_line(aes(y=trans_x, colour="trans_x")) +
      geom_line(aes(y=trans_y, colour="trans_y")) +
      geom_line(aes(y=trans_z, colour="trans_z")) +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Translation", x="Frame", y="Position (cm)")
    #print(mergedTransPlot)
    mergedRotPlot = ggplot(data=merged, aes(x=frame)) +
      geom_line(aes(y=true_rot_x, colour="rot_x"), linetype="dashed") +
      geom_line(aes(y=true_rot_y, colour="rot_y"), linetype="dashed") +
      geom_line(aes(y=true_rot_z, colour="rot_z"), linetype="dashed") +
      geom_line(aes(y=rot_x, colour="rot_x")) +
      geom_line(aes(y=rot_y, colour="rot_y")) +
      geom_line(aes(y=rot_z, colour="rot_z")) +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Rotation", x="Frame", y="Angle (radians)")
    #print(mergedRotPlot)
    mergedQuatPlot = ggplot(data=merged, aes(x=frame)) +
      geom_line(aes(y=true_q1, colour="q1"), linetype="dashed") +
      geom_line(aes(y=true_q2, colour="q2"), linetype="dashed") +
      geom_line(aes(y=true_q3, colour="q3"), linetype="dashed") +
      geom_line(aes(y=true_q4, colour="q4"), linetype="dashed") +
      geom_line(aes(y=q1, colour="q1")) +
      geom_line(aes(y=q2, colour="q2")) +
      geom_line(aes(y=q3, colour="q3")) +
      geom_line(aes(y=q4, colour="q4")) +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Quaternion values: (q1, q2, q3) = rotation axis, q4 = cos(angle/2)", x="Frame", y="Quaternion values")
    #print(mergedQuatPlot)
    grid.arrange(mergedTransPlot, mergedRotPlot, mergedQuatPlot, nrow=3, main=textGrob(paste("Object pose: ", pose_file, " (solid) vs. ground truth: ", fcurves_file, " (dashed)", sep=""), vjust=1, gp=gpar(fontface="bold", cex=1.1)))
  
    # * Create separate fcurve and pose plots [deprecated: too verbose; use merged plots]
    # ** Plot fcurves (NOTE For fcurves, Y, Z are flipped: y = -location_2, z = location_1)
    fcurvesTransPlot = ggplot(data=fcurves, aes(x=frame)) +
      geom_line(aes(y=location_0*fcurves_trans_scale, colour="trans_x"), linetype="dashed") +
      geom_line(aes(y=-location_2*fcurves_trans_scale, colour="trans_y"), linetype="dashed") +
      geom_line(aes(y=location_1*fcurves_trans_scale, colour="trans_z"), linetype="dashed") +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Object pose", x="Frame", y="Position (units?)")
    fcurvesRotPlot <- ggplot(data=fcurves, aes(x=frame)) +
      geom_line(aes(y=rotation_euler_0*fcurves_rot_scale, colour="rot_x"), linetype="dashed") +
      geom_line(aes(y=-rotation_euler_2*fcurves_rot_scale, colour="rot_y"), linetype="dashed") +
      geom_line(aes(y=rotation_euler_1*fcurves_rot_scale, colour="rot_z"), linetype="dashed") +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Object pose", x="Frame", y="Angle (radians)")
  
    # ** Plot pose
    poseTransPlot <- ggplot(data=pose, aes(x=frame)) +
      geom_line(aes(y=trans_x, colour="trans_x")) +
      geom_line(aes(y=trans_y, colour="trans_y")) +
      geom_line(aes(y=trans_z, colour="trans_z")) +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Object pose", x="Frame", y="Position (units?)")
    poseRotPlot <- ggplot(data=pose, aes(x=frame)) +
      geom_line(aes(y=rot_x, colour="rot_x")) +
      geom_line(aes(y=rot_y, colour="rot_y")) +
      geom_line(aes(y=rot_z, colour="rot_z")) +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Object pose", x="Frame", y="Angle (radians)")
  
    # ** Combine fcurves and pose together
    combinedTransPlot = fcurvesTransPlot +
      geom_line(data=pose, aes(x=frame, y=trans_x, colour="trans_x")) +
      geom_line(data=pose, aes(x=frame, y=trans_y, colour="trans_y")) +
      geom_line(data=pose, aes(x=frame, y=trans_z, colour="trans_z")) +
      guides(colour=guide_legend(title="Property")) +
      labs(title="Object pose", x="Frame", y="Position (units?)")
    #print(combinedTransPlot)
  }
  
  return(list(ground_truth=ground_truth, pose=pose, merged=merged))
}

plotPose2 <- function(base_filename, pose_dir=default_pose_dir) {
  return(plotPose(fcurves_file=paste(base_filename, "_fcurves.dat", sep=""), pose_file=paste(base_filename, ".dat", sep=""), pose_dir))
}

plotPose3Multi <- function(base_filename, pose_dir=default_pose_dir, step.pattern=asymmetricP0, window.type="itakura") {
  res_by_fps = list()
  for(fps in fps_rates) {
    res = plotPose(fcurves_file=paste(base_filename, "_fcurves.dat", sep=""), pose_file=paste(base_filename, "_", fps, "fps.dat", sep=""), pose_dir)
    
    # Compare pose and ground_truth to find out error
    trans = res$merged[,c('trans_x', 'trans_y', 'trans_z')]
    true_trans = res$merged[,c('true_trans_x', 'true_trans_y', 'true_trans_z')]
    dtw_trans = dtw(trans, true_trans, step.pattern=step.pattern, window.type=window.type)
    rot = res$merged[,c('rot_x', 'rot_y', 'rot_z')]
    true_rot = res$merged[,c('true_rot_x', 'true_rot_y', 'true_rot_z')]
    dtw_rot = dtw(rot, true_rot, step.pattern=step.pattern, window.type=window.type)
    res$dtw_trans = dtw_trans
    res$dtw_rot = dtw_rot
    res_by_fps[[as.character(fps)]] = res
  }
  return(res_by_fps)
}

analyzeTestSets <- function(test_sets=all_test_sets, pose_dir=default_pose_dir) {
  # Header for output table
  cat("motion_axis")
  for(fps in fps_rates) {
    cat(paste("\tfps", fps, "_pos", "\tfps", fps, "_ori", sep=""))
  }
  cat("\n")
  
  # Loop over all tests
  res_by_test = list()
  dtw_dist = data.frame(motion=as.factor(NA), axis=as.factor(NA), fps=as.vector(NA, mode="numeric"), component=as.factor(NA), value=as.vector(NA, mode="numeric"))
  dtw_dist = dtw_dist[-1,]  # drop row of NAs
  for(base_filename in test_sets) {
    base_filename_parts = strsplit(base_filename, "-")[[1]]
    res = plotPose3Multi(base_filename, pose_dir)
    res_by_test[[base_filename]] = res
    
    # Extract normalized DTW distance for trans & rot
    cat(paste(base_filename_parts[2], base_filename_parts[3], sep="_"))  # skip 00- prefix
    for(fps in fps_rates) {
      dtw_dist = rbind(dtw_dist, data.frame(motion=base_filename_parts[2], axis=base_filename_parts[3], fps=fps, component="position", value=res[[as.character(fps)]]$dtw_trans$normalizedDistance))
      dtw_dist = rbind(dtw_dist, data.frame(motion=base_filename_parts[2], axis=base_filename_parts[3], fps=fps, component="orientation", value=res[[as.character(fps)]]$dtw_rot$normalizedDistance))
      
      cat(paste("\t", res[[as.character(fps)]]$dtw_trans$normalizedDistance, "\t", res[[as.character(fps)]]$dtw_rot$normalizedDistance, sep=""))
    }
    cat("\n")
  }
  return(list(res_by_test=res_by_test, dtw_dist=dtw_dist))
}

# Examples
#plotPose2("01-translate-X")  # default: fcurves relative to camera
#plotPose(fcurves_file="01-translate-X_fcurves.dat", pose_file="01-translate-X.dat")  # fcurves relative to camera
#plotPose(fcurves_file="01-translate-X_fcurves_abs.dat", pose_file="01-translate-X.dat")  # fcurves absolute, with Blender's coordinate origin as (0, 0, 0)
#plotPose3Multi("02-translate-Y", pose_dir="out/pose-multi_2014-01-06/")
#all_res = analyzeTestSets(pose_dir="out/pose-multi_2014-01-06/")
#all_res$dtw_dist  # DTW distances
#write.table(all_res$dtw_dist, file="dtw-results.dat", sep="\t", row.names=FALSE, quote=FALSE)  # write DTW values to file

# To plot DTW distances (components by motions):
#ggplot(all_res$dtw_dist, aes(x=fps, y=value, group=axis, colour=axis)) + geom_line() + facet_grid(component~motion, scale="free")  # raw, without proper labels
#ggplot(all_res$dtw_dist, aes(x=fps, y=value, group=axis, colour=axis)) + geom_line(lwd=1.2) + facet_grid(component~motion, scale="free") + labs(title="DTW distances for different motions and pose components", x="Input video frame rate (at 30 fps translate: 8 cm/s, rotate: 1.57 radians/s)", y="DTW distance (position: cm, orientation: radians)", colour="Motion axis") + theme(axis.title.x=element_text(vjust=-0.5), axis.title.y=element_text(vjust=0.25), title=element_text(vjust=1.0))
