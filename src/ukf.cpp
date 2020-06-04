#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <exception>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  /* hyperparameter */
  std_a_ = 3.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  /* hyperparameter */
  std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // Lots of initialization...
  is_initialized_ = false;
  time_us_ = 0;
  n_aug_ = 7;
  n_x_ = 5;
  weights_ = VectorXd(2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  lambda_ = 3 - n_aug_;
  x_.fill(0.0);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * Driver method for UKF.
   */

  // Handle first measurement
  if (!is_initialized_) {
    is_initialized_ = true;
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0),
            meas_package.raw_measurements_(1),
            0,
            0,
            0;
      /* hyperparameter */
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    } else {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      x_ << rho*cos(phi), rho*sin(phi), rho_dot, 0, 0;
      /* hyperparameter */
      P_ << std_radr_*std_radr_, 0, 0, 0, 0,
            0, std_radr_*std_radr_, 0, 0, 0,
            0, 0, std_radrd_*std_radrd_, 0, 0,
            0, 0, 0, std_radphi_*std_radphi_, 0,
            0, 0, 0, 0, 1;
    }
    time_us_ = meas_package.timestamp_;
    return;
  }

  double delta_t_ = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t_);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else {
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  // Generate augmented sigma points at current epoch
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

  // predicted state vector
  VectorXd xp_ = VectorXd(5);

  // predicted state covariance matrix
  MatrixXd Pp_ = MatrixXd(5, 5);

  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  MatrixXd second_term = sqrt(lambda_ + n_aug_) * A;

  for (int i=1; i<=n_aug_; ++i) {
    Xsig_aug.col(i) = x_aug + second_term.col(i - 1);
    Xsig_aug.col(i + n_aug_) = x_aug - second_term.col(i - 1);
  }

  // Predict sigma point
  for (int i=0; i<2*n_aug_+1; ++i) {
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    double px_p, py_p;
   
    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + (v/yawd) * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + (v/yawd) * (-cos(yaw + yawd * delta_t) + cos(yaw));
    } else {
        px_p = p_x + v * cos(yaw) * delta_t;
        py_p = p_y + v * sin(yaw) * delta_t;
    }

    px_p =  px_p + .5 * delta_t * delta_t * cos(yaw) * nu_a;
    py_p = py_p + .5 * delta_t * delta_t * sin(yaw) * nu_a;

    double v_p = v + delta_t * nu_a;
    double yaw_p = yaw + yawd * delta_t + .5 * delta_t * delta_t * nu_yawdd;
    double yawd_p = yawd + delta_t * nu_yawdd;
    
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // Get predicted mean and covariance
  // set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i=1; i<2*n_aug_+1; ++i) {
    weights_(i) = .5/(lambda_+n_aug_);
  }

  // predict state mean
  xp_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; ++i) {
      xp_ = xp_ + weights_(i) * Xsig_pred_.col(i);
  }

  Pp_.fill(0.0);
  // predict state covariance matrix
  for (int i=0; i<2*n_aug_+1; ++i) {
    VectorXd diff = Xsig_pred_.col(i) - xp_;
    while (diff(3)> M_PI) diff(3) -= 2.*M_PI;
    while (diff(3)<-M_PI) diff(3) += 2.*M_PI;
    Pp_ = Pp_ + weights_(i) * diff * diff.transpose();
  }
  x_ = xp_;
  P_ = Pp_;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int n_z = 2;

  //  - create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  //  - residual covariance matrix R
  MatrixXd R = MatrixXd(n_z,n_z);

  R.fill(0.0);
  R(0, 0) = std_laspx_*std_laspx_;
  R(1, 1) = std_laspy_*std_laspy_;

  for (int i=0; i<2*n_aug_+1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    // transform sigma points into measurement space
    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  UpdateSensorMeas(n_z, meas_package.raw_measurements_, Zsig, R, meas_package.sensor_type_);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   */

  int n_z = 3;

  //  - create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);

  //  - measurement noise covariance matrix R
  MatrixXd R = MatrixXd(n_z,n_z);

  R.fill(0.0);
  R(0, 0) = std_radr_*std_radr_;
  R(1, 1) = std_radphi_*std_radphi_;
  R(2, 2) = std_radrd_*std_radrd_;

  for (int i=0; i<2*n_aug_+1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double phi = Xsig_pred_(3, i);
    double phi_dot = Xsig_pred_(4, i);

    // transform sigma points into measurement space
    double ro = sqrt(px*px + py*py);
    double psi = atan2(py, px);
    double rod = (px*cos(phi)*v + py*sin(phi)*v)/ro;
    Zsig(0, i) = ro;
    Zsig(1, i) = psi;
    Zsig(2, i) = rod;
  }

  UpdateSensorMeas(n_z, meas_package.raw_measurements_, Zsig, R, meas_package.sensor_type_);
}

void UKF::UpdateSensorMeas(int n_z, VectorXd &z, MatrixXd &Zsig, MatrixXd &R, MeasurementPackage::SensorType type) {
  //  - mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //  - measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //  - cross-corellation matrix Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Initial matrics
  z_pred.fill(0.0);
  S.fill(0.0);
  Tc.fill(0.0);

  for (int i=0; i<2*n_aug_+1; ++i) {
    // calculate mean predicted measurement
    z_pred = z_pred + weights_(i) * Zsig.col(i);     
  }

  // calculate innovation covariance matrix S
  for (int i=0; i<2*n_aug_+1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    if (type == MeasurementPackage::RADAR) {
      // angle normalization
      while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R;

  // Update
  //  - calculate cross correlation matrix
  for (int i=0; i<2*n_aug_+1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    if (type == MeasurementPackage::RADAR) {
      while (z_diff(1)> M_PI) z_diff(1) -= 2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1) += 2.*M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
 
  //  - calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //  - update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  // angel normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}