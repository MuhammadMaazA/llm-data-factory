import { useEffect } from "react";

const Confetti = () => {
  useEffect(() => {
    import("canvas-confetti").then((confetti) => {
      confetti.default({
        particleCount: 150,
        spread: 70,
        origin: { y: 0.6 },
      });
    });
  }, []);
  return null;
};

export default Confetti;
