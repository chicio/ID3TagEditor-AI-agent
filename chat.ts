import * as readline from 'readline';
import { chatModel } from './chat-model';

const askQuestion = async () => {
    const chat = await chatModel()
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });    

    rl.question('Question: ', async (question) => {
        if (question.toLowerCase() === '/stop') {
            console.log('Good bye!!');
            rl.close();
            return;
        }

        try {
            console.log('\n-------------------');
            let response = await chat.invoke({ question: question })

            console.log('\nAnswer:', response.answer);
            console.log('\n-------------------\n');
            
            askQuestion();
        } catch (error) {
            console.error('❌ Errore:', error);
            askQuestion();
        }
    });
};

askQuestion();

