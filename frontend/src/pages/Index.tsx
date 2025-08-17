import Navbar from "@/components/landing/Navbar";
import Hero from "@/components/landing/Hero";
import Workflow from "@/components/landing/Workflow";
import TechAndSetup from "@/components/landing/Sections";
import LiveDemo from "@/components/landing/LiveDemo";
import Results from "@/components/landing/Results";
import FAQ from "@/components/landing/FAQ";
import FutureWork from "@/components/landing/FutureWork";
import Footer from "@/components/landing/Footer";

const Index = () => {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <section id="workflow"><Workflow /></section>
        <TechAndSetup />
        <LiveDemo />
        <section id="results"><Results /></section>
        <FAQ />
        <FutureWork />
      </main>
      <Footer />
    </>
  );
};

export default Index;
