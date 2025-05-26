import background from '@assets/images/background.png';
import { CameraScreen } from '@components/CameraView';
import { API_BASE_URL } from '@constants/config';
import { useCameraPermission } from '@hooks/useCameraPermission';
import axios from 'axios';
import * as FileSystem from 'expo-file-system';
import React, { useState } from 'react';
import { ActivityIndicator, Button, Image, ImageBackground, StyleSheet, Text, View, TouchableOpacity } from 'react-native';

const Home = () => {
  const hasPermission = useCameraPermission();
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const uploadImage = async (uri: string) => {
    setLoading(true);
    const fileInfo = await FileSystem.getInfoAsync(uri);

    const formData = new FormData();
    formData.append('file', {
      uri: fileInfo.uri,
      name: 'photo.jpg',
      type: 'image/jpeg',
    } as any);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const { skin_type, confidence, message } = response.data;
      setResult(skin_type);
      setConfidence(confidence);
      setMessage(message);
    } catch (err) {
      setResult('Failed to detect');
      setMessage('Terjadi kesalahan saat prediksi');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (hasPermission === null) return <Text>Requesting camera permission...</Text>;
  if (hasPermission === false) return <Text>Permission denied</Text>;

  return (
    <ImageBackground source={background} style={styles.container}>
      <Text style={styles.header}>Skinalyze</Text>

      {!imageUri ? (
        // Render camera view
        <CameraScreen onPictureTaken={(uri) => { setImageUri(uri); uploadImage(uri); }} />
      ) : (
        // Render image preview and result
        <View style={styles.resultContainer}>
            <Image source={{ uri: imageUri }} style={styles.preview} />
          
          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              setImageUri(null);
              setResult(null);
            }}
          >
            <Text style={styles.buttonText}>Retake</Text>
          </TouchableOpacity>
        </View>
      )}

      {loading && <ActivityIndicator size="large" color="#555" />}
      {result && !loading && (
        <View style={styles.resultBackground}>
          <Text style={styles.result}>Skin Type: {result}</Text>
          {confidence !== null && (
            <Text style={styles.result}>Confidence: {(confidence * 100).toFixed(2)}%</Text>
          )}
        </View>
      )}

    </ImageBackground>
  );
};

export default Home

const styles = StyleSheet.create({
  container: { flex: 1, padding: 10, justifyContent: 'center' },
  header: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFF8E1',
    textAlign: 'center',
    paddingVertical: 12,
    borderTopLeftRadius: 40,
    borderTopRightRadius: 40,
    marginBottom: 10,
  },
  preview: {
    width: '85%',
    height: 400,
    borderRadius: 20,
    overflow: 'hidden',
    marginVertical: 20,
    backgroundColor: '#000',
    alignSelf: 'center',
    borderWidth: 2,
    borderColor: '#FFF8E1',
    resizeMode: 'cover', // changed from 'contain' to 'cover'
  },
  result: {
    fontSize: 20,
    textAlign: 'center',
    marginTop: 10,
    color: '#333',
  },
  button: {
    backgroundColor: '#FC9B78',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 12,
    alignSelf: 'center',
    marginBottom: 30,
  },
  buttonText: {
    color: '#FFF8E1',
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultBackground:{
    backgroundColor: '#FC9B78',
    borderRadius: 18,
    padding: 18,
    marginHorizontal: 24,
    marginTop: 10,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 3,
  }
});