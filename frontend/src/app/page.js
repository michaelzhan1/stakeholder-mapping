"use client"

import TextStakeholderExtract from "@/components/TextStakeholderExtract";
import RLRunner from "@/components/RLRunner";
import FullPipeline from "@/components/FullPipeline";
import LLMInfoModal from "@/components/LLMInfoModal";

import { useState } from "react";


export default function Home() {
  const [useFullPipeline, setUseFullPipeline] = useState(false);

  return (
    <>
      {useFullPipeline ?
        <button onClick={() => setUseFullPipeline(false)} className="text-blue-500 underline mb-5">Use Separate Components</button>
        :
        <button onClick={() => setUseFullPipeline(true)} className="text-blue-500 underline mb-5">Use Full Pipeline</button>
      }
      <LLMInfoModal />
      {useFullPipeline ?
        <FullPipeline />
        :
        <>
          <TextStakeholderExtract />
          <RLRunner />
        </>
      }
    </>
  );
}
