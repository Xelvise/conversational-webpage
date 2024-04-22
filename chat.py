import streamlit as st, time
from utils import *
from streamlit_float import float_init, float_parent

# app config
st.set_page_config(page_title="Chat with webpages", page_icon="🤖")
st.title("Chat with webpages")
cont = st.container()

# if 'url_entered' not in st.session_state:
#     st.session_state.url_entered = False

# sidebar
with st.sidebar:
    # create a text_input widget for the URL
    url = st.text_input("Enter a Webpage URL", placeholder="https://www.example.com")
    if st.button("Submit URL"):
        if url:
            st.session_state.vector_store, st.session_state.page_title = get_vectorstore_from_url(url)
            page_title = st.session_state.page_title.split('•')[0]
            cont.caption(f'### TITLE: {page_title}')
            memory.clear()
        else:
            st.toast('Enter a URL to commence chat')

    float_init()
    with st.container():
        if st.button('Clear Chats', use_container_width=True):
            if os.path.exists('chat_history.json'):
                os.remove('chat_history.json')
                del st.session_state.messages
                st.rerun()
            else:
                pass
        float_parent("bottom: 2%;")


# initialize chat history
if 'messages' not in st.session_state:
    load_conversation_history()

# display chat messages from history on app rerun
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar='img/420bb535f6bd4081a8cb4308e95f9768.jpg'):
            st.write(msg["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(msg["role"]):
            st.write(msg["content"], unsafe_allow_html=True)
            
# accept user input
if prompt := st.chat_input(placeholder='Enter your prompt...'):
    st.session_state.messages.append({"role":"user", "content":prompt})     # Add user msg to chat history
    st.chat_message('user', avatar='img/420bb535f6bd4081a8cb4308e95f9768.jpg').write(prompt)   # Display user msg in chat msg container
    st.toast('thinking...')#; time.sleep(4); st.toast('give me few secs...')
    try:
        response = get_response(prompt, st.session_state.vector_store)
    except AttributeError:
        st.chat_message('assistant').write("My apologies! I couldn't retrieve any information. Please enter a Web URL.")
        st.session_state.messages.append({"role":"assistant", "content":"My apologies! I couldn't retrieve any information. Please enter a Web URL."})
    else:
        st.chat_message('assistant').write(response)
        st.session_state.messages.append({"role":"assistant", "content":response})
    save_conversation_history()



