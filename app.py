import streamlit as st
from dotenv import load_dotenv 
import defines 



def main():

    load_dotenv()

    st.set_page_config(page_title='ai_chat_pdfs', page_icon=':rocket:')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    promt = st.chat_input('Asking question ...')


    with st.sidebar:
        st.subheader('Your documents:')

        pdf_docs = st.file_uploader('Upload your PDFs here:', accept_multiple_files=True)

        if st.button('Start processing'):
            with st.spinner('Processing...'):
                # Get text
                raw_text = defines.get_pdf_text(pdf_docs)

                # Chunk text
                chunked_text = defines.get_text_chunked(raw_text)

                # Create database
                vectorstore = defines.get_vectorstore(chunked_text)
                
                # Create conversation
                st.session_state.conversation = defines.get_conversation_chain(vectorstore)

                st.markdown('Done processing')

    if promt:
        defines.handle_user_input(st.session_state.conversation, promt)

if __name__=="__main__":
    main()