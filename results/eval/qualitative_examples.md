# Qualitative Flip Examples
Hard-bucket LLM examples that were correctly identified as LLM at P0 but misclassified as Human at P2 under simplified paraphrasing.

---

## Example 1 — `llm_0000148`

**char-TF-IDF detector:** P(LLM) = 0.858 (P0) → 0.410 (P2)   _correct → wrong_

**Length:** 151 → 113 words   **Mean word length:** 6.28 → 4.81 chars   **TTR:** 0.656 → 0.637

### P0 (original Llama-3.1-8B output)

> Botany is the scientific study of plants, including their structure, growth, development, reproduction, metabolism, evolution, and classification. It encompasses a wide range of disciplines, from plant anatomy and morphology to physiology, biochemistry, genetics, ecology, and systematics. Botanists investigate various aspects of plant biology, such as photosynthesis, respiration, nutrient uptake, and response to environmental stimuli. They also study the diversity of plant species, including their classification, distribution, and interactions with other organisms in ecosystems. The field of botany has numerous applications in agriculture, horticulture, forestry, medicine, and conservation, among others. Botanists use a variety of techniques, including microscopy, spectroscopy, and molecular biology, to analyze plant tissues, cells, and molecules. Their research contributes to our understanding of the complex relationships between plants and their environment, as well as the development of new technologies and products derived from plants. As a result, botany continues to be an essential field of study in modern science.

### P1 (one round of simplified paraphrasing)

> The study of plants is called botany. It's a scientific field that looks at how plants grow, develop, and reproduce. Botany also explores how plants make their own food, how they respond to their environment, and how they interact with other living things. Botanists study the different parts of plants, like their cells and molecules, and how they work together. They also look at the many different types of plants and how they're distributed around the world. Botany has many practical uses, such as helping farmers grow crops, developing new medicines, and conserving forests. Botanists use special tools and techniques, like microscopes and molecular biology, to learn more about plants. Their research helps us understand the world around us and can lead to new discoveries and innovations.

### P2 (two rounds of simplified paraphrasing)

> The study of plants is called botany. It's a science that looks at how plants grow, develop, and make babies. Botany also studies how plants make their own food and how they react to their surroundings. Botanists examine the different parts of plants, like their tiny building blocks, and how they work together. They also study the many different types of plants and where they can be found. Botany has many practical uses, such as helping farmers grow food, creating new medicines, and protecting forests. Botanists use special tools and techniques to learn more about plants, and their research helps us understand the world and can lead to new discoveries and ideas.

---

## Example 2 — `llm_0000116`

**char-TF-IDF detector:** P(LLM) = 0.700 (P0) → 0.326 (P2)   _correct → wrong_

**Length:** 147 → 124 words   **Mean word length:** 5.66 → 4.68 chars   **TTR:** 0.694 → 0.645

### P0 (original Llama-3.1-8B output)

> Axiology is the branch of philosophy that deals with the study and evaluation of values. It encompasses various aspects of value theory, including the nature, origin, and significance of values in human life. Axiologists examine the concepts of good, bad, right, wrong, beautiful, and ugly to understand how they are perceived and applied across different cultures and societies. The field also explores the relationship between values and other philosophical disciplines such as ethics, aesthetics, and epistemology. Axiological inquiry can be divided into two main areas: descriptive axiology, which seeks to describe and classify existing values, and normative axiology, which aims to establish standards for evaluating and justifying values. By analyzing and comparing different value systems, axiologists aim to provide a deeper understanding of the complex and multifaceted nature of human values, their role in shaping individual and collective behavior, and their impact on society as a whole.

### P1 (one round of simplified paraphrasing)

> The study of values is called axiology. It's a part of philosophy that tries to understand what values are, where they come from, and why they're important in our lives. Axiologists look at how people in different cultures and societies think about things like good and bad, right and wrong, beautiful and ugly. They also explore how values relate to other areas of philosophy, such as ethics, art, and knowledge. There are two main ways to study values: one is to describe and classify the values that already exist, and the other is to figure out what standards should be used to judge and justify values. By comparing different value systems, axiologists try to understand how values shape our behavior and affect society as a whole.

### P2 (two rounds of simplified paraphrasing)

> The study of values is called axiology, which is a part of philosophy that tries to understand what values are and why they're important in our lives. Axiologists look at how people in different cultures and societies think about things like good and bad, right and wrong, and what is beautiful or ugly. They also explore how values relate to other areas of philosophy, such as ethics, art, and knowledge. There are two main ways to study values: one is to describe and classify the values that already exist, and the other is to figure out what standards should be used to judge and justify values. By comparing different value systems, axiologists try to understand how values affect society and shape our behavior.

---

## Example 3 — `llm_0000207`

**char-TF-IDF detector:** P(LLM) = 0.644 (P0) → 0.342 (P2)   _correct → wrong_

**word-TF-IDF detector:** P(LLM) = 0.655 (P0) → 0.436 (P2)

**Length:** 150 → 132 words   **Mean word length:** 5.14 → 4.31 chars   **TTR:** 0.68 → 0.644

### P0 (original Llama-3.1-8B output)

> Al Capone was an American gangster and businessman who rose to prominence during the Prohibition era in the United States. Born on January 17, 1899, in Brooklyn, New York, he moved to Chicago with his family at a young age and quickly became involved in organized crime. Capone's early career included working as a bouncer and enforcer for various gangs before eventually becoming the leader of the Outfit, a powerful organized crime syndicate that dominated Chicago's underworld during the 1920s. He made his fortune through various illicit activities, including bootlegging, prostitution, and extortion, and became notorious for his brutal tactics and flamboyant lifestyle. Capone's reign was marked by violent conflicts with rival gangs and law enforcement, but he managed to evade prosecution until 1931, when he was finally convicted of tax evasion and sentenced to 11 years in prison. He died on January 25, 1947, at the age of 48, due to complications from syphilis.

### P1 (one round of simplified paraphrasing)

> Al Capone was a famous American gangster and businessman who lived during a time when the sale of alcohol was banned in the United States. He was born in Brooklyn, New York in 1899 and moved to Chicago with his family when he was young. Capone quickly got involved in organized crime and became a leader of a powerful gang that controlled Chicago's underworld in the 1920s. He made money by doing things like selling bootlegged liquor, running brothels, and extorting money from people. He was known for being violent and living a flashy lifestyle. Capone had many run-ins with the law, but he wasn't caught until 1931, when he was charged with not paying his taxes and sentenced to 11 years in prison. He died in 1947 at the age of 48 due to complications from a long-term illness.

### P2 (two rounds of simplified paraphrasing)

> Al Capone was a well-known American gangster and businessman who lived during a time when the sale of alcohol was banned in the United States. He was born in Brooklyn, New York in 1899 and moved to Chicago with his family when he was young. Capone quickly got involved in organized crime and became a leader of a powerful gang in Chicago's underworld in the 1920s. He made money by selling bootlegged liquor, running brothels, and taking money from people by force. He was known for being violent and living a flashy lifestyle. Capone had many problems with the law, but he wasn't caught until 1931, when he was charged with not paying his taxes and sentenced to 11 years in prison. He died in 1947 at the age of 48 due to health problems.

---
