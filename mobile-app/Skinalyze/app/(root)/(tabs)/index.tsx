import { Link } from "expo-router";
import { Text, View } from "react-native";

export default function Index() {
  return (
    <View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
        <Link href="./home">Home</Link>
        <Link href="./result">Result</Link>
        <Link href="./camera">Take a Picture</Link>
    </View>
  );
}
