"use client"

const POSITION_DEF = "Position (-1: Opposed, 0: Neutral, 1: Supportive): Position refers to the stakeholder's stance on the issue relative to the primary stakeholder, based on their actions.\n" +
"- Opposed (-1): The stakeholder actively works against the objectives of the primary stakeholder.\n" +
"- Neutral (0): The stakeholder does not take a definitive stance relative to the primary stakeholder.\n" +
"- Supportive (1): The stakeholder actively supports the objectives of the primary stakeholder.";

const POWER_DEF = "Power (0: Low, 1: Medium, 2: High): Power refers to the degree of control or influence a stakeholder has over the outcome of the negotiation. Sources of power may include legal authority, financial resources, political leverage, or social capital. Consider the stakeholder's access to these resources and how directly they can affect the final decision.\n" +
"- ⁠Low (0): The stakeholder has minimal or no control over the negotiation's outcome. They might be unable to mobilize resources or people effectively, have no formal authority, or be easily overruled. Even if they are engaged in the process, their influence is limited to providing input, with little to no impact on decisions.\n" +
"⁠- Medium (1): The stakeholder holds some influence over the negotiation but does not have decisive control. They may influence others' decisions or introduce changes, but their power is indirect or dependent on coalition-building. While they can sway opinions or set agendas, they are not the final decision-maker and require cooperation from others to effect change.\n" +
"⁠- High (2): The stakeholder has decisive control or a significant degree of leverage over the negotiation's outcome. They possess critical resources or authority and can dictate terms, veto decisions, or influence others to a degree that the negotiation cannot proceed effectively without their consent.";

const KNOWLEDGE_DEF = "Knowledge (0: Not Aware, 1: Limited, 2: Extensive): Knowledge refers to the stakeholder's level of awareness and actionable insights.\n" +
"- Not Aware (0): Lacks any awareness of the relevant issue.\n" +
"- Limited (1): Has some level of awareness but lacks sufficient information to effectively participate in or influence the negotiation.\n" +
"- Extensive (2): Possesses relevant insights and understanding that significantly affect the negotiation outcomes.";

const URGENCY_DEF = "Urgency (0: Low, 1: High): Urgency refers to the stakeholder's subjective perception of the time-sensitivity of the issue.\n" +
"- Low (0): The issue can be addressed on a flexible timeline without immediate consequences in the stakeholder's view.\n" +
"- High (1): The stakeholder perceives the issue as requiring immediate action to prevent significant negative outcomes (e.g., legal deadlines, financial losses, or operational disruptions).";

const LEGITIMACY_DEF = "Legitimacy (0: Not Legitimate, 1: Legitimate): Legitimacy refers to whether a stakeholder has the right or standing to be involved in the negotiation.\n" +
"- Not Legitimate (0): The stakeholder should not be involved in the negotiation.\n" +
"- Legitimate (1): The stakeholder has the relevance and standing to meaningfully participate in the negotiation.";

export const EXTRACTION_PROMPT_CORE = "Given the following text data, identify the key stakeholders involved in the negotiation and infer their attributes based on the definitions provided below. You do not have prior knowledge of the ground truth values, so you must analyze the text and make educated estimates about the stakeholders' values for Power, Urgency, Knowledge, Position, and Legitimacy.\n" +
"\n" +
"For each stakeholder, determine the following attributes and present the values in CSV format. Your prediction must be from the available options for each attribute, make your best guess if necessary. Do not use any markdown formatting, and only have the CSV data in raw text. Ensure a stakeholder name does not have any commas, for the sake of the CSV. Each row should look like the following:\n" +
"\n" +
"<Stakeholder name>,<Position: (-1, 0, or 1)>,<Power: (0, 1, or 2)>,<Urgency: (0 or 1)>,<Knowledge: (0, 1, or 2)>,<Legitimacy: (0 or 1)>\n" +
"\n" +
"Do not output any additional text. Use only the information implied in the text to make these determinations.\n" +
"\n" +
"Definitions:\n" +
`${POWER_DEF}\n` +
`${URGENCY_DEF}\n` +
`${KNOWLEDGE_DEF}\n` +
`${LEGITIMACY_DEF}\n` +
`${POSITION_DEF}\n` +
"\n";

export default function LLMInfoModal() {
  return (
    <>
      <div className='fixed inset-0 bg-gray-800 bg-opacity-50 flex items-center justify-center'>
        <div className='flex justify-center bg-gray-500 w-1/2 h-1/2'>
          asdfasdf
        </div>
      </div>
    </>
  )
}