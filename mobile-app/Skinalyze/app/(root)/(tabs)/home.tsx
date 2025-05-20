import React, { useState } from 'react';
import { View, Image, Text, Button, ActivityIndicator, StyleSheet } from 'react-native';
import * as FileSystem from 'expo-file-system';
import axios from 'axios';
import { API_BASE_URL } from '@constants/config';
import { useCameraPermission } from '@hooks/useCameraPermission';
import { CameraScreen } from '@components/CameraView';

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
    <View style={styles.container}>
      {!imageUri ? (
        <CameraScreen onPictureTaken={(uri) => { setImageUri(uri); uploadImage(uri); }} />
      ) : (
        <>
          <Image source={{ uri: imageUri }} style={styles.preview} />
          <Button title="Retake" onPress={() => { setImageUri(null); setResult(null); }} />
        </>
      )}

      {loading && <ActivityIndicator size="large" />}
      {result && !loading && (
          <View>
            <Text style={styles.result}>Skin Type: {result}</Text>
            {confidence !== null && (
              <Text style={styles.result}>Confidence: {(confidence * 100).toFixed(2)}%</Text>
            )}
            {message && <Text style={styles.result}>Message: {message}</Text>}
          </View>
      )}

    </View>
  );
};

export default Home

const styles = StyleSheet.create({
  container: { flex: 1, padding: 10, justifyContent: 'center' },
  preview: { width: '100%', height: 400, resizeMode: 'contain', marginVertical: 10 },
  result: { fontSize: 20, textAlign: 'center', marginTop: 10 },
});