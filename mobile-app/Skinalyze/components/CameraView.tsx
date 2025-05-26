import {
  CameraCapturedPicture,
  CameraView,
  useCameraPermissions,
} from 'expo-camera';
import React, { useRef } from 'react';
import {
  StyleSheet,
  View,
  TouchableOpacity,
  Text,
  Dimensions,
} from 'react-native';

type Props = {
  onPictureTaken: (uri: string) => void;
};

export const CameraScreen: React.FC<Props> = ({ onPictureTaken }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo: CameraCapturedPicture =
        await cameraRef.current.takePictureAsync();
      onPictureTaken(photo.uri);
    }
  };

  if (!permission?.granted) {
    return (
      <View style={styles.permissionContainer}>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Request Camera Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.wrapper}>
      <View style={styles.cameraContainer}>
        <CameraView ref={cameraRef} style={styles.camera} facing="front" />
      </View>

      <TouchableOpacity style={styles.button} onPress={takePicture}>
        <Text style={styles.buttonText}>Take Picture</Text>
      </TouchableOpacity>
    </View>
  );
};

const { width } = Dimensions.get('window');
const previewHeight = width * 1.4; // buat rasio potret

const styles = StyleSheet.create({
  wrapper: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  cameraContainer: {
    width: width * 0.9,
    height: previewHeight,
    borderRadius: 30,
    overflow: 'hidden',
    marginVertical: 20,
    backgroundColor: '#000',
  },
  camera: {
    width: '100%',
    height: '100%',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#EFEBD6',
  },
  button: {
    backgroundColor: '#4B4A3F',
    paddingVertical: 12,
    paddingHorizontal: 30,
    borderRadius: 16,
    marginTop: 10,
  },
  buttonText: {
    color: '#FFF8E1',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
