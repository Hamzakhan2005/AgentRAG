import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function IntroScreen({ onComplete }) {
  const [progress, setProgress] = useState(0);
  const [done, setDone] = useState(false);

  useEffect(() => {
    // Simulate loading progress
    const steps = [
      { target: 30, delay: 200, speed: 18 },
      { target: 65, delay: 0, speed: 25 },
      { target: 85, delay: 0, speed: 40 },
      { target: 100, delay: 200, speed: 20 },
    ];

    let current = 0;
    let stepIndex = 0;

    const runStep = () => {
      if (stepIndex >= steps.length) return;
      const step = steps[stepIndex];
      setTimeout(() => {
        const interval = setInterval(() => {
          current += 1;
          setProgress(current);
          if (current >= step.target) {
            clearInterval(interval);
            stepIndex++;
            runStep();
            if (current >= 100) {
              setTimeout(() => setDone(true), 500);
            }
          }
        }, step.speed);
      }, step.delay);
    };

    runStep();
  }, []);

  useEffect(() => {
    if (done) {
      setTimeout(onComplete, 900);
    }
  }, [done, onComplete]);

  // Circle grows from center to fill screen
  const circleSize = done ? "250vmax" : "65vmin";

  return (
    <AnimatePresence>
      {!done || true ? (
        <motion.div
          className="intro-overlay"
          initial={{ opacity: 1 }}
          animate={{ opacity: done ? 0 : 1 }}
          transition={{ duration: 0.8, ease: [0.76, 0, 0.24, 1] }}
          style={{ pointerEvents: done ? "none" : "all" }}
        >
          {/* Background oversized text */}
          <div className="intro-bg-text">AGENT RAG</div>

          {/* Expanding circle */}
          <motion.div
            className="intro-circle"
            animate={{
              width: circleSize,
              height: circleSize,
            }}
            transition={{
              duration: done ? 0.7 : 0.6,
              ease: [0.76, 0, 0.24, 1],
            }}
            style={{ width: "65vmin", height: "65vmin" }}
          >
            <motion.div
              className="intro-circle-inner"
              animate={{ opacity: done ? 0 : 1 }}
              transition={{ duration: 0.3 }}
            >
              <div className="intro-logo">Intelligence</div>
              <div className="intro-title">
                Agent
                <br />
                RAG
              </div>
              <div className="intro-percent">{progress}%</div>
              <div className="intro-bar">
                <motion.div
                  className="intro-bar-fill"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.1 }}
                />
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
