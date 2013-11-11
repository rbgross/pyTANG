# TANG: Comparing output pose data with ground-truth for virtual videos
# Run script: source("scripts/analyze-pose.r")

# Libraries
library(ggplot2)
library(grid)
library(gridExtra)
library(orientlib)  # for converting between Euler angles and quaternions

# Script parameters
## Files and directories
fcurves_dir = "res/videos/virtual/"
pose_dir = "out/pose/"

## Data scaling and transformation
fcurves_trans_scale = 100  # fcurves are in meters
pose_trans_scale = 0.5  # observed pose data is roughly in 0.5 cms (TODO verify this)

## Misc. constants
two_pi = 2 * pi

# Pose plotting (and analysis) function
plotPose <- function(fcurves_file, pose_file) {
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
  
  return(list(ground_truth=ground_truth, pose=pose, merged=merged))
}

plotPose2 <- function(base_filename) {
  return(plotPose(fcurves_file=paste(base_filename, "_fcurves.dat", sep=""), pose_file=paste(base_filename, ".dat", sep="")))
}

# Examples
#plotPose2("01-translate-X")  # default: fcurves relative to camera
#plotPose(fcurves_file="01-translate-X_fcurves.dat", pose_file="01-translate-X.dat")  # fcurves relative to camera
#plotPose(fcurves_file="01-translate-X_fcurves_abs.dat", pose_file="01-translate-X.dat")  # fcurves absolute, with Blender's coordinate origin as (0, 0, 0)
