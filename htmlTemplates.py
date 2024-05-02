css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #D0E0E3
}
.chat-message.bot {
    background-color: #B6D7A8
}
.chat-message .avatar {
  width: 100%;
}
.chat-message .avatar img {
  max-width: 38px;
  max-height: 38px;
  border-radius: 30%;
  object-fit: cover;
}
.chat-message .message {
  width: 100%;
  padding: 0 1.5rem;
  color: #000;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="Designer.png" alt="AI Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="renga_profile.jpg" alt="User Avatar">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
