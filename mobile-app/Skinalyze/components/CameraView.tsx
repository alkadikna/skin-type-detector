import React, { useRef } from 'react';
import { StyleSheet, View, Button } from 'react-native';
import { CameraView, useCameraPermissions, CameraCapturedPicture } from 'expo-camera';

type Props = {
  onPictureTaken: (uri: string) => void;
};

export const CameraScreen: React.FC<Props> = ({ onPictureTaken }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo: CameraCapturedPicture = await cameraRef.current.takePictureAsync();
      onPictureTaken(photo.uri);
    }
  };

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <Button title="Request Camera Permission" onPress={requestPermission} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="front" />
      <Button title="Take Picture" onPress={takePicture} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
});
