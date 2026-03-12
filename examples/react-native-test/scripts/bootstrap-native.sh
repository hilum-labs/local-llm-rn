#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$(mktemp -d)"
APP_NAME="LocalLLMTest"
RN_VERSION="0.83.0"
IOS_DEPLOYMENT_TARGET="16.0"
ANDROID_COMPILE_SDK="35"
ANDROID_TARGET_SDK="35"
ANDROID_MIN_SDK="33"
APP_BUNDLE_ID="com.hilum.localllmtest"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [ ! -d "$APP_DIR/ios" ] || [ ! -d "$APP_DIR/android" ]; then
  echo "Generating bare React Native native projects for ${APP_NAME}..."
  npx @react-native-community/cli init "$APP_NAME" \
    --directory "$TMP_DIR/$APP_NAME" \
    --version "$RN_VERSION" \
    --skip-install

  rm -rf "$APP_DIR/ios" "$APP_DIR/android"
  cp -R "$TMP_DIR/$APP_NAME/ios" "$APP_DIR/ios"
  cp -R "$TMP_DIR/$APP_NAME/android" "$APP_DIR/android"

  if [ -f "$TMP_DIR/$APP_NAME/babel.config.js" ] && [ ! -f "$APP_DIR/babel.config.js" ]; then
    cp "$TMP_DIR/$APP_NAME/babel.config.js" "$APP_DIR/babel.config.js"
  fi
fi

if [ -f "$APP_DIR/ios/Podfile" ]; then
  perl -0pi -e "s/platform :ios, min_ios_version_supported/platform :ios, '${IOS_DEPLOYMENT_TARGET}'/g; s/platform :ios, '[0-9.]+'/platform :ios, '${IOS_DEPLOYMENT_TARGET}'/g" "$APP_DIR/ios/Podfile"
fi

find "$APP_DIR/ios" -name project.pbxproj -print0 | while IFS= read -r -d '' file; do
  perl -0pi -e "s/IPHONEOS_DEPLOYMENT_TARGET = [0-9.]+;/IPHONEOS_DEPLOYMENT_TARGET = ${IOS_DEPLOYMENT_TARGET};/g; s/PRODUCT_BUNDLE_IDENTIFIER = [^;]+;/PRODUCT_BUNDLE_IDENTIFIER = ${APP_BUNDLE_ID};/g" "$file"
done

find "$APP_DIR/android" \( -name '*.gradle' -o -name '*.gradle.kts' \) -print0 | while IFS= read -r -d '' file; do
  perl -0pi -e "s/compileSdk\\s*=\\s*[0-9]+/compileSdk = ${ANDROID_COMPILE_SDK}/g; s/compileSdkVersion\\s*=\\s*[0-9]+/compileSdkVersion = ${ANDROID_COMPILE_SDK}/g; s/compileSdkVersion\\s+[0-9]+/compileSdkVersion ${ANDROID_COMPILE_SDK}/g; s/targetSdk\\s*=\\s*[0-9]+/targetSdk = ${ANDROID_TARGET_SDK}/g; s/targetSdkVersion\\s*=\\s*[0-9]+/targetSdkVersion = ${ANDROID_TARGET_SDK}/g; s/targetSdkVersion\\s+[0-9]+/targetSdkVersion ${ANDROID_TARGET_SDK}/g; s/minSdk\\s*=\\s*[0-9]+/minSdk = ${ANDROID_MIN_SDK}/g; s/minSdkVersion\\s*=\\s*[0-9]+/minSdkVersion = ${ANDROID_MIN_SDK}/g; s/minSdkVersion\\s+[0-9]+/minSdkVersion ${ANDROID_MIN_SDK}/g; s/namespace\\s*=\\s*\"[^\"]+\"/namespace = \"${APP_BUNDLE_ID}\"/g; s/applicationId\\s*=\\s*\"[^\"]+\"/applicationId = \"${APP_BUNDLE_ID}\"/g; s/applicationId\\s+\"[^\"]+\"/applicationId \"${APP_BUNDLE_ID}\"/g" "$file"
done

echo "Bare React Native native projects are ready."
